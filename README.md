# NCDD: Nearest Centroid Distance Deficit
### The official Pytorch implementation of NCDD: Nearest Centroid Distance Deficit for Out-Of-Distribution Detection in Gastrointestinal Vision paper available at [here](https://arxiv.org/abs/2412.01590).

![Clinical Overview](https://github.com/bhattarailab/NCDD/blob/main/intro.png)

**Landscape of clinical procedures in gastrointestinal vision**. **_Orange_**: Unassisted, a doctor has to assess all patients’ data tediously and redundantly. **_Blue_**: Artificial Intelligence can help in the classification of known or seen diseases but makes misleading assumptions and often overconfident predictions on images when it faces real-world examples consisting of examples that it had never seen. **_Green_**: A combination of human intervention and OOD enabled AI method to improve efficacy in the current scenario, where a specialist intervenes to correct any unseen or unknown instances that the AI model is uncertain in classifying.


### Update
- Paper accepted at MIUA 2025
- Nominated for best paper award

## Setup
The code to train the models is available in the directory model_training. For OOD calculation and evaluation the code is available in the directory ood with the implementation of our method and other OOD methods we tested.

```bash
git clone https://github.com/bhattarailab/NCDD
cd NCDD
```

### Environment
The required packages of the environment we used to conduct experiments are listed in requirements.txt
```bash
pip install -r requirements.txt
```

### Datasets
The two publicly available datasets on which OOD methods are evaluated are [Kvasirv2](https://datasets.simula.no/kvasir/) and [Gastrovision](https://github.com/DebeshJha/GastroVision).


## Usage
To run the ood detection, do the following:

For KVASIR dataset:
```bash
cd ood
bash demo_kvasir.sh
```
For Gastrovision dataset:
```bash
cd ood
bash demo_gastro.sh
```
## Contact
- [Sandesh Pokhrel](mailto:sandesh.pokhrel@naamii.org.np)
- [Sanjay Bhandari](mailto:sanjay.bhandari@naamii.org.np)
- [Binod Bhattarai](mailto:binod.bhattarai@abdn.ac.uk)
