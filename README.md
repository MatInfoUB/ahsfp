# Atomic Hirshfeld Surface Fingerprint

This software package takes an arbitary crystal structure to predict material properties.

![Architecture](figs/architecture.png)

#### Table of Contents

- [Prerequisites](#prerequisites)
- [How to Use](#how-to-use)
- [How to Cite](#how-to-cite)
- [Data](#data)
- [Authors](#authors)
- [License](#license)

##### Prerequisites

- [Pandas](#https://pandas.pydata.org/)
- [Keras](#https://keras.io/) (with [Tensorflow](#https://www.tensorflow.org/) backend)
- [Scikit-Learn](https://scikit-learn.org/stable/)


##### How to use:

###### TODO

Training the model

###### Using a pre-trained model:

Download the package using the following code:

`git clone https://github.com/arpanisi/mof_single_atom_hs.git`

Use conda to create an environment as:

`conda create -n mofpy python=3.7 scikit-learn keras tensorflow`

This creates a conda environment along with installing the prerequisites. Activate the environemnt by:

`conda activate mofpy`

Navigate to the folder `/mof_single_atom_hs` and type:

`python predict.py`

Alternately it can be used by one of the following codes:

`python predict.py --parameter old_lattice` and
`python predict.py --parameter new_lattice`. The output of both the commands give the following output

<table align="center">
<td align="center">
<img src="https://github.com/arpanisi/ahsfp/blob/master/figs/old_lattice.png" alt="Old Lattice Parameter" width="450px" />
</td>
<td align="center">
<img src="https://github.com/arpanisi/ahsfp/blob/master/figs/new_lattice.png" alt="Old Lattice Parameter" width="450px" />
</td>
</table>


#### How to cite

TODO

#### Data

TODO

#### Authors

This software is written by Arpan Mukherjee. Data Collection and analysis by Logan Williams. 
Arpan and Logan were advised by Prof. Krishna Rajan

#### License

released under the MIT License




