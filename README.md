# Analysis of _Komagataella phaffii_ at extremely slow growth rates. 

## This repository contains the code for analysis of:
## [Characterising the metabolic rewiring of extremely slow growing _Komagataella phaffii_, Coltman _et_al_, 2024](https://doi.org/10.1111/1751-7915.14386) and [Protein production dynamics and physiological adaptation of recombinant _Komagataella phaffii_ at near-zero growth rates, Rebnegger, _et al_, 2024](https://doi.org/10.1186/s12934-024-02314-3) 

## Regression model for [Rebnegger _et al_, 2024](https://doi.org/10.1186/s12934-024-02314-3)

The code to run the regression model of [Rebnegger _et al_, 2024](https://doi.org/10.1186/s12934-024-02314-3) can be found at [RebneggerRetentostatRegression.ipynb](scripts/RebneggerRetentostatRegression.ipynb). The code uses functions from [RegressionFuncs.py](scripts/RegressionFuncs.py). The output (.csv files and plots) of the regression analysis is saved in [results/Rebnegger2023](results/Rebnegger2023). The same functions are used in the analysis of [Coltman _et al_, 2024](https://doi.org/10.1111/1751-7915.14386); however, the model is extended, replacing the growth-rate:productivity relationship and static death rate parameter, with an alternative formulation.

## Analysis from [Coltman _et al_, 2024](https://doi.org/10.1111/1751-7915.14386)
### 1. Downloading and reproducing results

The results presented in [Coltman _et al_, 2024](https://doi.org/10.1111/1751-7915.14386) can be reproduced by cloning this repository, defining a conda environment and executing the included scripts. [iMT1026-NZ](results/iMT1026-NZ.xml) was generated during this analysis and contains all of the generated biomass equations described in the manuscript.

Most notebooks do not require a long time to execute, apart from the two notebooks that use flux sampling, which required approximately 8 hours of compute time using 50 cores ([ThinningAnalysis.ipynb](scripts/ThinningAnalysis.ipynb) and [SamplingAnalysis.ipynb](scripts/SamplingAnalysis.ipynb)). To avoid having to repeat the flux sampling, the chains that were used in the publication  ([thinning analysis](results/sampling/thinning_test) and [sampling analysis](results/sampling)) are included in this repository. Additionally, the notebooks using these sampled chains can be executed without sampling. Sampling can can be performed by uncommenting the cells in each respective notebook ([ThinningAnalysis.ipynb](scripts/ThinningAnalysis.ipynb) and [SamplingAnalysis.ipynb](scripts/SamplingAnalysis.ipynb)). 

The code to download, define the environment and execute the notebooks is:

```
cd Kphaffii_NearZero
conda env create -f kphaffii_nz.yaml
conda activate memo3.7
jupyter nbconvert --execute --ExecutePreprocessor.kernel_name=memo37 --to notebook --inplace LipidInvestigation.ipynb DataProcessing.ipynb RegressionModel.ipynb GasExchanges.ipynb ModelUpdate.ipynb GenerateDynamicComposition.ipynb  ModelIntegration.ipynb ThinningAnalysis.ipynb SamplingAnalysis.ipynb EscherPlotting.ipynb
```

### 2. Notebook overview

The biomass compositions are generated in [DataProcessing.ipynb](scripts/DataProcessing.ipynb) using data from [Rebnegger et al. 2024](https://doi.org/10.1186/s12934-024-02314-3), in addition to using the Ceramide content predicted in [LipidInvestigation.ipynb](scripts/LipidInvestigation.ipynb). 

The regression model is defined and fitted to cultivation data from [Rebnegger et al. 2024](https://doi.org/10.1186/s12934-024-02314-3) in [RetentostatRegression.ipynb](scripts/RetentostatRegression.ipynb). The gas exchange rates are determined in [GasExchanges.ipynb](scripts/GasExchanges.ipynb) and combined with the regression fits to determine the parameters used in constraint based modelling.

The starting model, [iMT1026v3]() is downloaded and some modifications are made to it in [ModelUpdate.ipynb](scripts/ModelUpdate.ipynb). 

The biomass equations are generated from the biomass compositions using three different methods in [GenerateDynamicComposition.ipynb](scripts/GenerateDynamicComposition.ipynb).

The cultivation data, updated model and biomass equations are used to determine the energetic parameters, predict growth rates and generate pFBA distributions in [ModelIntegration.ipynb](scripts/ModelIntegration.ipynb). These pFBA flux distributions are plotted using Escher in [EscherPlotting.ipynb](scripts/EscherPlotting.ipynb).

Flux sampling is used in this publication. The first notebook determines an optimal thinning parameter to achieve convergence with the OptGp sampler, and is determined at one growth rate (0.1 h<sup>-1</sup>) [ThinningAnalysis.ipynb](scripts/ThinningAnalysis.ipynb). Using this thinning factor, in [SamplingAnalysis.ipynb](scripts/SamplingAnalysis.ipynb) fluxes were sampled at all other growth rates and with different growth-rate specific biomass equations. 

### 3. Figures and tables in publication
Figures and tables for this analsysis can be found as follows:

[Figure 1](results/plots/RetentostatRegression.png)
[Figure 2a](results/plots/Mu_vs_qATP_GAM.png)
[Figure 2b](results/plots/MuComparisonRelative_WithCO2_AdjustedATPM.png)
[Figure 3](results/plots/Ridge_6x6_Gly&PPP.png)
[Figure 4, based on escher map](results/maps/DerivedC0.1vs16.9_GlucoseNormalizedMap.html)
[Figure 5](results/plots/Ridge_6x6_ETC.png)
[Figure 6](results/plots/Mu_vs_Cofactors.png)

[Figure S1](results/plots/5ComponentFit.png)
[Figure S2](results/plots/Log2RelativeStoich.png)
[Figure S3](results/plots/DynBiomassScaled.png)
[Figure S4](results/plots/ATPReduction_AltNADHDehydrogenase.png)

[Table S2](results/dataframes/cultivation_data/StatsRetentostat.png)
[Table S3](results/dataframes/biomass/CarbohydrateComposition.csv)
[Table S4](results/dataframes/BiomassCompositions.csv)
[Table S5](results/dataframes/AllBioMacros.csv)
[Table S6](results/dataframes/fluxes/95%CIFluxRatiosInteresting.csv)
[Table S7](results/dataframes/MajorCofactorsProductionBySubsystem.csv)
[Tables S8](results/dataframes/sampling/Thinning_AllStats.csv)

For an extended version of Table S6, with 95% CI of fluxes of all reactions, at all growth rates with all biomass equations, [see here](results/dataframes/fluxes/95%CIFluxRatios.csv)

#### 4. DOI
Our most recent Zenodo DOI generated for the journal submission is: 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10148579.svg)](https://doi.org/10.5281/zenodo.10148579)


#### 5. Contact
- benjamin.coltman@boku.ac.at
- juergen.zanghellini@univie.ac.at

#### 6. Citation
If you are using our work, please cite us at:

Coltman, B.L., Rebnegger, C., Gasser, B. & Zanghellini, J. (2024) Characterising the metabolic rewiring of extremely slow growing Komagataella phaffii. Microbial Biotechnology, 17, e14386. Available from: [https://doi.org/10.1111/1751-7915.14386](https://doi.org/10.1111/1751-7915.14386)
