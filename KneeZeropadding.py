import ismrmrd
import numpy as np
import matplotlib.pyplot as plt


# Finder filen
path = r"C:/Data/2dknee.h5"

# Tildeler filen som en array
dset = ismrmrd.Dataset(path, "dataset")

# Læser information om encoding fra xml filen
header = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
enc = header.encoding[0]

kx = enc.encodedSpace.matrixSize.x # Definerer størrelsen af matricen i kx retning - 352
ky_size = enc.encodedSpace.matrixSize.y # Definerer størrelsen af matricen i ky retning - 202

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

def fillkspace():
    """
    Der bliver kørt igennem alle acquisitions (samples)
    og fylder række for række for den valgte slice og den valgte coil
    Hvis den slice vi kigger på ikke er vores valgte slice, siger den continue (går videre til næste iteration)
    """
    for i in range(dset.number_of_acquisitions()):
        acq = dset.read_acquisition(i)  # Vælger hvilken acquisition vi kigger på

        if acq.idx.slice != slice_number:
            continue

        # kspace_encode_step_1 fortæller os i hvilket row den enkelte acquisition hører til.
        row = acq.idx.kspace_encode_step_1
        # For at filtrere noget støj (eller kalibrering) væk sørger vi for kun at kigge indenfor vores matrixsize
        # hvis vi får en row værdi som er negativ eller større end de 202, ser vi bort fra den
        if row < 0 or row >= ky_size:
            continue

        """
        data svarer til acquisition data i vores acq array, som indeholder alt dataet for hver sample
        # hver acquisition har coil number og tilsvarende samples.
        # vi sætter coil number til en specifik værdi og får dermed kun vores samples på linjen.
        """
        line = acq.data[coil_number, :]

        """
        n = min(n,kx)
        Vælger den mindste værdi af enten kx eller line.shape[0].
        nx er bredden af vores k-space som defineret tidligere.
        line.shape[0] kigger på dimensionen af det line array, hvilket er bredden altså antal indgange i arrayen
        Det gør vi for at begrænse vores opfyldning af k-space til den størrelse, 
        som vi har defineret den til at være. (Ser bort fra eventuel støj osv.)
        """
        n = min(kx, line.shape[0])

        """
        Den line af kx værdier, som vi har bestemt bliver tildelt hvert row, da row bliver indekseret.
        Bredden af matricen kspace og bredden af vektoren line bliver begge begrænset af den samme værdi n.
        Så en line svarer til en linje i kspace.
        """
        kspace[row, :n] = line[:n]
    return kspace

def transform(kspace):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace)))

image = transform(fillkspace())

# mangler kommentarer på følgende funktion
def samplingmask(ky_size, acceleration_factor=7, center_fraction=0.04, seed=42):
    """
    Genererer en tilfældig undersamplings maske med et fully sampled centrum.

    ky_size:             Antal rækker i k-space
    acceleration_factor: Hvor hurtigt vi sampler (R=4 betyder vi beholder 1/4 af linjerne)
    center_fraction:     Andelen af k-space centrum som altid er fuldt samplet (0.08 = 8%)
    seed:                Seed til tilfældig sampling, så vi får samme maske hver gang
    """
    np.random.seed(seed)
    mask = np.zeros(ky_size, dtype=bool)

    # Beregner start og slut indeks for centrum af k-space
    center_start = ky_size // 2 - int(ky_size * center_fraction) // 2
    center_end = ky_size // 2 + int(ky_size * center_fraction) // 2
    mask[center_start:center_end] = True

    # Beregner hvor mange linjer vi vil sample udenfor centrum
    n_center = mask.sum()
    n_total = ky_size // acceleration_factor
    n_random = n_total - n_center
    print(f'Rækker beholdt: {n_total}') # printer hvor mange rækker vi beholder ud af de 202

    # Sampler tilfældigt fra de resterende linjer udenfor centrum
    outside_center = np.where(~mask)[0]
    chosen = np.random.choice(outside_center, size=int(n_random), replace=False)
    mask[chosen] = True

    return mask

def undersampling(kspace):
    mask = samplingmask(ky_size)
    kspace_undersampled = kspace.copy()  # undersampler en kopi af kspace
    kspace_undersampled[~mask, :] = 0
    return kspace_undersampled

kspace_undersampled = undersampling(kspace)
image_undersampled = transform(kspace_undersampled)

def l2reconstruction():
    """


    """

    return

def MeanSquareError(image_ref, image_recon):
    """

    :param image_ref: Reference image ("perfekte" billede)
    :param image_recon: Reconstructed image
    :return: Returnerer den absolutte fejl mellem det "perfekte" billede og rekonstruktionen
    men med en skalering på den højeste reference værdi, da det er meget høje værdier
    """
    ref = np.abs(image_ref)
    recon = np.abs(image_recon)

    return np.mean(np.square(ref - recon))

def RelativeMeanSquareError(image_ref, image_recon):
    """

    :param image_ref: Reference image ("perfekte" billede)
    :param image_recon: Reconstructed image
    :return: Returnerer de relative fejl mellem det "perfekte" billede og rekonstruktionen
    men med en skalering, da det er meget høje værdier
    """
    ref = np.abs(image_ref)
    recon = np.abs(image_recon)

    return np.sum(np.square(ref - recon)) / np.sum(np.square(ref))

print(f'Zerofill - Absolute error: {MeanSquareError(image, image_undersampled): .1f}')
print(f'Zerofill - Relative error: {RelativeMeanSquareError(image, image_undersampled): .4f}')

# laver subplots, så vi kan få dem på samme figure
fig, axes = plt.subplots(2,2,figsize=(10,5)) # ændre ncols ud fra hvor mange rekonstruktioner vi vil vise

# Plotter logaritmisk da der er rigtig stor forskel på værdierne.
# Plotter magnitude af kspace
# cmap='gray' gør plottet grayscale
axes[0,0].imshow(np.log(np.abs(kspace)+1E-09), cmap='gray') # OBS: kan eventuelt tilføje meget småt tal til np.log..., da log(0) ikke er defineret
axes[0,0].set_title("Fully sampled k-space (log scale)")

# Plotter fully sampled image efter fourier transform
axes[1,0].imshow(np.abs(image),cmap='gray')
axes[1,0].set_title("Fully sampled image reconstruction")

axes[0,1].imshow(np.log(np.abs(kspace_undersampled)+1E-09), cmap='gray')
axes[0,1].set_title("Undersampled k-space (log scale)")

axes[1,1].imshow(np.abs(image_undersampled),cmap='gray')
axes[1,1].set_title("Undersampled image reconstruction")

#axes[0,2].imshow(np.log(np.abs(kspace_undersampled)+1E-09),cmap='gray') # skal der laves en ny kspace af det rekonstrueret billede?
#axes[0,2].set_title("Undersampled k-space (log scale)")

#axes[1,2].imshow(np.abs(image_reconstruction),cmap='gray')
#axes[1,2].set_title("L2 reconstructed image")

plt.tight_layout()
plt.show()
