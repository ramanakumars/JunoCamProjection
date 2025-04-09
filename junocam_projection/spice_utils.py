import os
import glob
from bs4 import BeautifulSoup
import requests
import re
import fnmatch
import tqdm

BASEURL = 'https://naif.jpl.nasa.gov/pub/naif/JUNO/kernels/'


def fetch_kernels_from_https(path: str, pattern: str) -> list[str]:
    """Fetch kernels from URL matching a given pattern

    :param path: URL at which to search for kernels (will parse HTML links in this URL)
    :param pattern: file-name pattern to search for relevant links

    :return: list of files that match the given pattern
    """
    with requests.get(path) as response:
        soup = BeautifulSoup(response.text, 'html.parser')
    kernels_all = [a['href'] for a in soup.find('pre').find_all('a')]
    files = fnmatch.filter(kernels_all, pattern)
    basefolder = os.path.split(os.path.normpath(path))[-1]
    return [os.path.join(basefolder, file) for file in files]


def fetch_kernels_from_disk(path: str, pattern: str) -> list[str]:
    """Fetch kernels from local path matching a given pattern

    :param path: path to the folder at which to search for kernels
    :param pattern: file-name pattern to search for relevant links

    :return: list of files that match the given pattern
    """
    return [file.replace(path, "") for file in sorted(glob.glob(os.path.join(path, pattern)))]


def check_and_download_kernels(kernels: list[str], KERNEL_DATAFOLDER: str) -> list[str]:
    """Check whether the list of kernels are in the local directory and download them if needed.

    :param kernels: list of kernel filenames (relative to the root URL)
    :param KERNEL_DATAFOLDER: path to the local kernel directory to store downloaded kernels

    :return: list of local paths to the kernels
    """
    kernel_fnames = []
    for kernel in kernels:
        if not os.path.exists(os.path.join(KERNEL_DATAFOLDER, kernel)):
            download_kernel(kernel, KERNEL_DATAFOLDER)
        kernel_fnames.append(os.path.join(KERNEL_DATAFOLDER, kernel))

    return kernel_fnames


def download_kernel(kernel: str, KERNEL_DATAFOLDER: str) -> None:
    """Download a given kernel to the local folder

    :param kernel: URL path to the kernel
    :param KERNEL_DATAFOLDER: root folder to store the downloaded kernel
    """
    link = os.path.join(BASEURL, kernel)
    file_name = os.path.join(KERNEL_DATAFOLDER, kernel)
    with open(file_name, "wb") as f:
        print("Downloading %s" % file_name)
        response = requests.get(link, stream=True)
        total_length = response.headers.get('content-length')

        if total_length is None:  # no content length header
            f.write(response.content)
        else:
            total_length = int(total_length)
            with tqdm.tqdm(total=total_length, unit='B', unit_scale=True, dynamic_ncols=True, unit_divisor=1024, ascii=True, desc=f'Downloading {kernel}') as pbar:
                for data in tqdm.tqdm(response.iter_content(chunk_size=4096)):
                    f.write(data)
                    pbar.update(len(data))


def get_kernels(KERNEL_DATAFOLDER: str, start_utc: float, offline: bool = False) -> list[str]:
    """Fetch all relevant kernels for JunoCam at a given time

    :param KERNEL_DATAFOLDER: path to the local kernel directory to store downloaded kernels
    :param start_utc: the spacecraft clock time in start_utc
    :param offline: use the kernels stored locally (saves time by not scraping the NAIF servers)

    :return: list of local paths to the kernels
    """
    if not os.path.exists(KERNEL_DATAFOLDER):
        os.mkdir(KERNEL_DATAFOLDER)

    for folder in ['ik', 'ck', 'spk', 'pck', 'fk', 'lsk', 'sclk']:
        if not os.path.exists(os.path.join(KERNEL_DATAFOLDER, folder)):
            os.mkdir(os.path.join(KERNEL_DATAFOLDER, folder))

    if offline:
        print("Fetching kernels from disk")
        iks = fetch_kernels_from_disk(KERNEL_DATAFOLDER, "ik/juno_junocam_v*.ti")
        cks = fetch_kernels_from_disk(KERNEL_DATAFOLDER, "ck/juno_sc_*.bc")
        spks1 = fetch_kernels_from_disk(KERNEL_DATAFOLDER, "spk/spk_*.bsp")
        spks2 = fetch_kernels_from_disk(KERNEL_DATAFOLDER, "spk/jup*.bsp")
        spks3 = fetch_kernels_from_disk(KERNEL_DATAFOLDER, "spk/de*.bsp")
        spks4 = fetch_kernels_from_disk(KERNEL_DATAFOLDER, "spk/juno_struct*.bsp")
        pcks = fetch_kernels_from_disk(KERNEL_DATAFOLDER, "pck/pck*.tpc")
        fks = fetch_kernels_from_disk(KERNEL_DATAFOLDER, "fk/juno_v*.tf")
        sclks = fetch_kernels_from_disk(KERNEL_DATAFOLDER, "sclk/JNO_SCLKSCET.*.tsc")
        lsks = fetch_kernels_from_disk(KERNEL_DATAFOLDER, "lsk/naif*.tls")
    else:
        print("Fetching kernels from NAIF server")
        iks = fetch_kernels_from_https(BASEURL + "ik/", "juno_junocam_v*.ti")
        cks = fetch_kernels_from_https(BASEURL + "ck/", "juno_sc_*.bc")
        spks1 = fetch_kernels_from_https(BASEURL + "spk/", "spk_*.bsp")
        spks2 = fetch_kernels_from_https(BASEURL + "spk/", "jup*.bsp")
        spks3 = fetch_kernels_from_https(BASEURL + "spk/", "de*.bsp")
        spks4 = fetch_kernels_from_https(BASEURL + "spk/", "juno_struct*.bsp")
        pcks = fetch_kernels_from_https(BASEURL + "pck/", "pck*.tpc")
        fks = fetch_kernels_from_https(BASEURL + "fk/", "juno_v*.tf")
        sclks = fetch_kernels_from_https(BASEURL + "sclk/", "JNO_SCLKSCET.*.tsc")
        lsks = fetch_kernels_from_https(BASEURL + "lsk/", "naif*.tls")

    year, month, day = start_utc.split("-")
    yy = year[2:]
    mm = month
    dd = day[:2]

    intdate = int("%s%s%s" % (yy, mm, dd))

    kernels = []

    # find the ck and spk kernels for the given date
    ckpattern = r"juno_sc_rec_([0-9]{6})_([0-9]{6})\S*"
    nck = 0
    for ck in cks:
        fname = os.path.basename(ck)
        groups = re.findall(ckpattern, fname)
        if len(groups) == 0:
            continue
        datestart, dateend = groups[0]

        if (int(datestart) <= intdate) & (int(dateend) >= intdate):
            kernels.append(ck)
            nck += 1

    """ use the predicted kernels if there are no rec """
    if nck == 0:
        print("Using predicted CK")
        ckpattern = r"juno_sc_pre_([0-9]{6})_([0-9]{6})\S*"
        for ck in cks:
            fname = os.path.basename(ck)
            groups = re.findall(ckpattern, fname)
            if len(groups) == 0:
                continue
            datestart, dateend = groups[0]

            if (int(datestart) <= intdate) & (int(dateend) >= intdate):
                kernels.append(ck)
                nck += 1

    spkpattern = r"spk_rec_([0-9]{6})_([0-9]{6})\S*"
    nspk = 0
    for spk in spks1:
        fname = os.path.basename(spk)
        groups = re.findall(spkpattern, fname)
        if len(groups) == 0:
            continue
        datestart, dateend = groups[0]

        if (int(datestart) <= intdate) & (int(dateend) >= intdate):
            kernels.append(spk)
            nspk += 1

    """ use the predicted kernels if there are no rec """
    if nspk == 0:
        print("Using predicted SPK")
        spkpattern = r"spk_pre_([0-9]{6})_([0-9]{6})\S*"
        for spk in spks1:
            fname = os.path.basename(spk)
            groups = re.findall(spkpattern, fname)
            if len(groups) == 0:
                continue
            datestart, dateend = groups[0]

            if (int(datestart) <= intdate) & (int(dateend) >= intdate):
                kernels.append(spk)
                nspk += 1

    assert nck * nspk > 0, "Kernels not found for the given date range!"

    # load the latest updates for these
    kernels.append(iks[-1])
    kernels.append(spks2[-1])
    kernels.append(spks3[-1])
    kernels.append(spks4[-1])
    kernels.append(pcks[-1])
    kernels.append(fks[-1])
    kernels.append(sclks[-1])
    kernels.append(lsks[-1])
    kernels.append("spk/juno_rec_orbit.bsp")
    kernels.append("spk/juno_pred_orbit.bsp")

    return check_and_download_kernels(kernels, KERNEL_DATAFOLDER)
