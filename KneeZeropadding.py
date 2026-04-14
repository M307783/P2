import ismrmrd
import numpy as np
import matplotlib.pyplot as plt


# Finder filen hurtigt
path = r"C:/Data/2dknee.h5"

# Tildeler filen som en array
dset = ismrmrd.Dataset(path, "dataset")

kx = 352
ky_size = 208

acq = dset.read_acquisition(0)
print(acq)

# Antallet af coils
# Kan også aflæses i xml
# ncoils = 16 # Udkommenteret da en matrice over dataen ville være 16 x 352 x 202 (how tf er en matrix 3d)

# Vælger 1 coil fremfor alle 16, så vi kan få en nx x ny matrix
# Indekset bestemmer hvilken af de 16 coils (0-15) vi vælger
coil_number = 0

# Vi kigger kun på en slice
# Indekset her bestemmer hvilken af de 28 slices (0-27) vi vælger
slice_number = 10 # Vælger 10 da det er et af de slices hvor man kan se mest på

# Generer en tom 2d matrix af størrelsen ny x nx.
# np.complex64 sørger for at hver indgang i matricen ligner og agerer som et komplekst tal
kspace = np.zeros((ky_size,kx),dtype=np.complex64)

"""
Der bliver kørt igennem alle acquisitions (samples)
og fylder række for række for den valgte slice og den valgte coil
Hvis den slice vi kigger på ikke er vores valgte slice, siger den continue (går videre til næste iteration)
"""
for i in range(dset.number_of_acquisitions()):
    acq = dset.read_acquisition(i) # Vælger hvilken acquisition vi kigger på

    # Bestemmer størrelsen af vores datasæt udfra hvor meget der er blevet samplet.
    # Værdierne kan aflæses i xml

    # Redefinerer ky til 202 fremfor 208
    ky = acq.idx.kspace_encode_step_1

    if acq.idx.slice != slice_number:
        continue

    """
    data svarer til acquisition data i vores acq array, som indeholder alt dataet for hver sample
    # hver acquisition har coil number og tilsvarende samples.
    # vi sætter coil number til en specifik værdi og får dermed kun vores samples på linjen.
    """
    line = acq.data[coil_number,:]

    """
    n = min(n,kx)
    Vælger den mindste værdi af enten kx eller line.shape[0].
    nx er bredden af vores k-space som defineret tidligere.
    line.shape[0] kigger på dimensionen af det line array, hvilket er bredden altså antal indgange i arrayen
    Det gør vi for at begrænse vores opfyldning af k-space til den størrelse, 
    som vi har defineret den til at være. (Ser bort fra eventuel støj osv.)
    """
    n =  min(kx,line.shape[0])

    """
    Den line af kx værdier, som vi har bestemt bliver tildelt hvert ky, da ky bliver indekseret.
    Bredden af matricen kspace og bredden af vektoren line bliver begge begrænset af den samme værdi n.
    Så en line svarer til en linje i kspace.
    """
    kspace[ky, :n] = line[:n]


# Plotter logaritmisk da der er rigtig stor forskel på værdierne.
# Plotter magnitude af kspace
# cmap='gray' gør plottet grayscale
plt.imshow(np.log(np.abs(kspace)), cmap='gray') # OBS: kan eventuelt tilføje meget småt tal til np.log..., da log(0) ikke er defineret
plt.title("K-space (log scale)")
plt.show()
