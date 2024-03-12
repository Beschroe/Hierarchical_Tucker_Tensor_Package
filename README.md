### Wissenschaftliche Grundlage
Das in diesem Repository enthaltene Python Package enthält eine Implementierung des hierarchischen Tuckerformats für Tensoren und beruht dabei
auf den beiden nachstehenden Arbeiten: 

- D. Kressner and C. Tobler, Algorithm 941, ACM Transactions on Mathematical Software, 40 (2014), pp. 1–22.
- L. Grasedyck, Hierarchical singular value decomposition of tensors, SIAM Journal on Matrix Analysis and Applications, 31 (2010), pp. 2029–2054.

### Requirements:
(Möglicherweise sind andere Versionen ebenfalls möglich. Getestet wurde das Package allerdings mit untenstehender Konfiguration)
- python = 3.10
- torch = 2.2.1
- numpy = 1.26.4

### Installation (zur Verwendung auf der CPU):
1) Erstellung eines neuen Conda environment
```
conda create -n ENV_NAME python=3.10
```
3) Wechseln in das neu erzeugte Environment
```
conda activate ENV_NAME
```
2) Installieren der vorausgesetzten Pakete
```
pip install torch==2.2.1 numpy==1.26.4
pip install numpy==1.26.4
```
3) Klonen von Hierarchical_Tucker_Tensor_Package in ein beliebiges Verzeichnis
```
cd PREFERED_DIRECTORY
git clone https://github.com/Beschroe/Hierarchical_Tucker_Tensor_Package.git
```
4) Installieren von Hierarchical_Tucker_Tensor_Package
```
pip install Hierarchical_Tucker_Tensor_Package/HTucker
```
5) Pruefen, ob Installation erfolgreich war
```
python
>>> from HTucker import HTTensor as htt
>>> x = htt.randn([2,3,4])
>>> x.full()
tensor([[[ 5.8944e-01, -1.4568e+00,  4.8685e-01, -1.0708e-01],
         [ 1.8261e-01, -1.9248e+00,  1.1095e+00, -8.7830e-01],
         [ 5.1478e-01, -4.1099e-02, -3.7586e-01,  6.1266e-01]],
        [[ 1.7282e-02, -4.2714e-02,  1.4274e-02, -3.1395e-03],
         [ 5.3540e-03, -5.6434e-02,  3.2530e-02, -2.5752e-02],
         [ 1.5093e-02, -1.2050e-03, -1.1020e-02,  1.7963e-02]]])
```
War die Installation erfolgreich, sollte obiges Beispiel funktionieren.
