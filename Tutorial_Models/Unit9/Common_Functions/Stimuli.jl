trial = [:target,:target,:target,:target,:target,
         :target,:target,:target,:target,:foil,:foil,:foil,:foil,
         :foil,:foil,:foil,:foil,:foil]
pep = [:lawyer,:captain,:hippie,:debutante,:earl,:hippie,
          :fireman,:captain,:hippie,:fireman,:captain,:giant,
          :fireman,:captain,:giant,:lawyer,:earl,:giant]
pla = [:store,:cave,:church,:bank,:castle,:bank,:park,
          :park,:park,:store,:store,:store,
          :bank,:bank,:bank,:park,:park,:park]

stimuli = [(trial=t,person=p,place=pl) for (t,p,pl) in zip(trial,pep,pla)]
