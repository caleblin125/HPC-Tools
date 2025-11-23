spack install boost@1.86.0 eigen@3.4.0 nlohmann-json@3.11.3 xerces-c@3.3.0 intel-tbb@2022.2.0

spack install root@6.34.04 +pythia8 +tbb ~xrootd ^pythia8@8.313
# spack install root@6.34.04 +jemalloc +pythia8 +tbb ~xrootd ^pythia8@8.313
spack install pythia8@8.313 +hepmc3 ^boost@1.86.0
spack install hepmc3@3.3.1 ~rootio ~protobuf +python +interfaces ^root@6.34.04
spack install lcio@2.22.5 ^root@6.34.04
spack install geant4@11.3.0 +threads +data ^boost@1.86.0 ^root@6.34.04 ^intel-tbb@2022.2.0 ^xerces-c@3.3.0
spack install dd4hep@1.32.1 +geant4units +hepmc3 +lcio +tbb ^boost@1.86.0 ^geant4@11.3.0 ^hepmc3@3.3.1 ^intel-tbb@2022.2.0 ^lcio@2.22.5 ^root@6.34.04 ^xerces-c@3.3.0

#NOT CORRECT VERSION
spack install acts@39.2.0 +odd +dd4hep +examples +geant4 +hempc3 +pythia8 +tbb ^boost@1.86.0 ^dd4hep@1.32.1 ^geant4@11.3.0 ^hepmc3@3.3.1 ^intel-tbb@2022.2.0 ^nlohmann-json@3.11.3 ^pythia8@8.313 ^root@6.34.04