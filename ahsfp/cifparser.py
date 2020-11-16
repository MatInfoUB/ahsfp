# PyTorch-free version of https://github.com/txie-93/cgcnn/tree/master/data
from pymatgen.core.structure import Structure
import os
import pandas as pd
import numpy as np
import warnings


class CifParser:

    def __init__(self, datadir, radius=8, max_num_nbr=12):

        self.radius, self.max_num_nbr = radius, max_num_nbr
        assert os.path.exists(datadir), 'Data directory does not exist!'
        id_prop_file = os.path.join(datadir, 'id_prop.csv')
        assert os.path.exists(id_prop_file), 'id_prop.csv does not exist!'
        atom_init_file = os.path.join(datadir, 'atom_init.json')
        assert os.path.exists(atom_init_file), 'atom_init.json does not exist!'

        self.id_prop_data = pd.read_csv(id_prop_file)
        self.atom_init_data = pd.read_json(atom_init_file)

        self.atom_types = self.atom_init_data.columns.to_list()
        self.list_cifs = [os.path.join(datadir, f) for f in os.listdir(datadir) if f.endswith('cif')]

    def gdf_expand(self, distances, dmin=0, dmax=10, step=0.1, tol=1e-3):

        filter = np.arange(dmin, dmax+step, step)
        gdf_dist = np.exp(-(distances[..., np.newaxis] - filter)**2 /
                      step**2)

        gdf_dist[gdf_dist < tol] = 0

        return gdf_dist

    def parse_data(self, list_of_cifs=None):

        if list_of_cifs is None:
            list_of_cifs = self.list_cifs

        return [self.parse_cif(ciffile=ciffile) for ciffile in list_of_cifs]

    def parse_cif(self, ciffile):

        crystal = Structure.from_file(ciffile)
        atom_fea = np.vstack([self.atom_init_data[c.specie.number]
                              for c in crystal])
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]

        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn('{} not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.'.format(ciffile))
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                               [self.radius + 1.] * (self.max_num_nbr -
                                                     len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1],
                                        nbr[:self.max_num_nbr])))
        nbr_fea_idx, nbr_fea = np.array(nbr_fea_idx), np.array(nbr_fea)
        nbr_fea = self.gdf_expand(nbr_fea, dmax=self.radius, step=0.2)

        return atom_fea, nbr_fea, nbr_fea_idx


