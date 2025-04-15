import pandas as pd
import torch
from tqdm import tqdm
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset
import os.path as osp
from typing import List, Optional, Callable, Union, Tuple

class MolecularDataset(InMemoryDataset):
    r"""Custom molecular dataset similar to QM9 structure but for specific molecular data.
    
    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk.
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset.
    """

    def __init__(self, root='.', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['41598_2019_47536_MOESM2_ESM.xlsx']

    @property
    def processed_file_names(self):
        return ['data.pt']
        
    def process(self) -> None:
        """Process raw data into PyG Data objects."""
        # First determine all atom and bond types in dataset
        atom_feature_vector_types = set()
        bond_feature_vector_types = set()
        
        raw_path = osp.join(self.raw_dir, self.raw_file_names[0])
        df = pd.read_excel(raw_path)
        self.smiles_list, self.target_list = df['Canonical_Smiles'].tolist(), df['class'].tolist()
        
        # Process molecules
        data_list = []
        for idx, (smiles, y) in enumerate(tqdm(zip(self.smiles_list, self.target_list), 
                                             total=len(self.smiles_list))):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            # Get atom features
            x = []
            for atom in mol.GetAtoms():
                atomic_number = atom.GetAtomicNum()
                formal_charge = atom.GetFormalCharge()
                
                hybridisation = atom.GetHybridization()
                hybridisation_dict = {
                    Chem.rdchem.HybridizationType.SP: 0,
                    Chem.rdchem.HybridizationType.SP2: 1,
                    Chem.rdchem.HybridizationType.SP3: 2,
                    Chem.rdchem.HybridizationType.SP3D: 3,
                    Chem.rdchem.HybridizationType.SP3D2: 4,
                }
                hybridisation_numerical = hybridisation_dict.get(hybridisation, 5)  # Default to 5 if not found
                
                is_aromatic = atom.GetIsAromatic()
                is_aromatic_numerical = 1 if is_aromatic else 0
                
                features = [
                    atomic_number,  # Atom type
                    formal_charge,  # Formal charge 
                    hybridisation_numerical,   # Hybridization
                    is_aromatic_numerical,  # Aromaticity
                ]
                
                atom_feature_vector_types.add(tuple(features))
                x.append(features)
                
            x = torch.tensor(x, dtype=torch.long)
            
            # Get edge indices and attributes
            edge_index = []
            edge_attr = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_index.append([i, j])
                edge_index.append([j, i])  # Undirected graph
                edge_attr.append([bond.GetBondTypeAsDouble()])
                edge_attr.append([bond.GetBondTypeAsDouble()])
                bond_feature_vector_types.add(bond.GetBondTypeAsDouble())

            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.long)

            # Create Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                smiles=smiles,
                idx=idx,
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        # Save processed data
        self.save(data_list, self.processed_paths[0])


class MolecularDatasetFactory:
    """Factory class to create molecular datasets from various data sources."""
    
    @staticmethod
    def from_csv(file_path: str, smiles_col: str = 'smiles', 
                 target_col: str = 'activity', root: str = '.', 
                 transform=None, pre_transform=None, pre_filter=None) -> MolecularDataset:
        """Create a dataset from a CSV file."""
        # Create a custom class extending MolecularDataset for this specific data
        class CustomMolecularDataset(MolecularDataset):
            @property
            def raw_file_names(self):
                return [osp.basename(file_path)]
                
            def process(self):
                raw_path = osp.join(self.raw_dir, self.raw_file_names[0])
                file_ext = osp.splitext(file_path)[1].lower()
                
                if file_ext == '.csv':
                    df = pd.read_csv(raw_path)
                elif file_ext in ['.xlsx', '.xls']:
                    df = pd.read_excel(raw_path)
                else:
                    raise ValueError(f"Unsupported file format: {file_ext}")
                    
                self.smiles_list = df[smiles_col].tolist()
                self.target_list = df[target_col].tolist()
                
                # Create same process as MolecularDataset but with custom column names
                # Process molecules
                data_list = []
                for idx, (smiles, y) in enumerate(tqdm(zip(self.smiles_list, self.target_list), 
                                                     total=len(self.smiles_list))):
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        continue

                    # Get atom features
                    x = []
                    for atom in mol.GetAtoms():
                        atomic_number = atom.GetAtomicNum()
                        formal_charge = atom.GetFormalCharge()
                        
                        hybridisation = atom.GetHybridization()
                        hybridisation_dict = {
                            Chem.rdchem.HybridizationType.SP: 0,
                            Chem.rdchem.HybridizationType.SP2: 1,
                            Chem.rdchem.HybridizationType.SP3: 2,
                            Chem.rdchem.HybridizationType.SP3D: 3,
                            Chem.rdchem.HybridizationType.SP3D2: 4,
                        }
                        hybridisation_numerical = hybridisation_dict.get(hybridisation, 5)
                        
                        is_aromatic = atom.GetIsAromatic()
                        is_aromatic_numerical = 1 if is_aromatic else 0
                        
                        features = [
                            atomic_number,  # Atom type
                            formal_charge,  # Formal charge 
                            hybridisation_numerical,  # Hybridization
                            is_aromatic_numerical,  # Aromaticity
                        ]
                        
                        x.append(features)
                        
                    x = torch.tensor(x, dtype=torch.long)
                    
                    # Get edge indices and attributes
                    edge_index = []
                    edge_attr = []
                    for bond in mol.GetBonds():
                        i = bond.GetBeginAtomIdx()
                        j = bond.GetEndAtomIdx()
                        edge_index.append([i, j])
                        edge_index.append([j, i])  # Undirected graph
                        edge_attr.append([bond.GetBondTypeAsDouble()])
                        edge_attr.append([bond.GetBondTypeAsDouble()])

                    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                    edge_attr = torch.tensor(edge_attr, dtype=torch.long)

                    # Create Data object
                    data = Data(
                        x=x,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        y=float(y),  # Ensure target is float
                        smiles=smiles,
                        idx=idx,
                    )

                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    data_list.append(data)

                # Save processed data
                self.save(data_list, self.processed_paths[0])
        
        # Import the data file to raw directory
        import os
        import shutil
        
        os.makedirs(osp.join(root, 'raw'), exist_ok=True)
        raw_path = osp.join(root, 'raw', osp.basename(file_path))
        if not osp.exists(raw_path):
            shutil.copyfile(file_path, raw_path)
            
        # Create and return the dataset
        return CustomMolecularDataset(root, transform, pre_transform, pre_filter)
    
    @staticmethod
    def from_smiles(smiles_list: List[str], target_list: List[float], 
                    root: str = '.', transform=None, 
                    pre_transform=None, pre_filter=None) -> MolecularDataset:
        """Create a dataset directly from SMILES strings and targets."""
        # Create a temporary CSV
        import tempfile
        import pandas as pd
        
        df = pd.DataFrame({
            'smiles': smiles_list,
            'activity': target_list
        })
        
        temp_dir = tempfile.mkdtemp()
        temp_file = osp.join(temp_dir, 'temp_data.csv')
        df.to_csv(temp_file, index=False)
        
        # Use the from_csv method
        return MolecularDatasetFactory.from_csv(
            temp_file, 'smiles', 'activity', 
            root, transform, pre_transform, pre_filter
        )