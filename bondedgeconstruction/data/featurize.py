from typing import List

from torch_geometric.data import Data, Batch

from bondedgeconstruction.data.bondgraph.to_bondgraph import to_bondgraph
from bondedgeconstruction.data.featurization.mol_to_data import mol_to_data
from bondedgeconstruction.data.featurization.smiles_to_3d_mol import smiles_to_3d_mol
import copy

def smiles_to_data(smiles: str) -> Data:

    mol = smiles_to_3d_mol(smiles)
    data = mol_to_data(mol)
    data = to_bondgraph(data)
    data.pos = None

    return data


def collate_with_circle_index(data_list: List[Data]) -> Batch:
    """
    Collates a list of Data objects into a Batch object.

    Args:
        data_list: a list of Data objects. Each Data object must contain `circle_index` attribute.
        k_neighbors: number of k consecutive neighbors to be used in the message passing step.

    Returns:
        Batch object containing the collate attributes from data objects, including `circle_index` collated
        to shape (total_num_nodes, max_circle_size).

    """
    batch1=copy.deepcopy(data_list)
    for b in batch1:
        b.x=b.x1

    batch1 = Batch.from_data_list(batch1, exclude_keys=['circle_index'])
    batch = Batch.from_data_list(data_list, exclude_keys=['circle_index'])
    batch.batch1=batch1.batch
    batch.edge_index1=batch1.edge_index1
    return batch
