import os
from bs4 import BeautifulSoup
import requests
import re
import fnmatch
import tqdm

BASEURL = 'https://naif.jpl.nasa.gov/pub/naif/JUNO/kernels/'


def fetch_kernels_from_https(path, pattern):
    with requests.get(path) as response:
        soup = BeautifulSoup(response.text, 'html.parser')
    kernels_all = [a['href'] for a in soup.find('pre').find_all('a')]
    return fnmatch.filter(kernels_all, pattern)


def check_and_download_kernels(kernels, KERNEL_DATAFOLDER):
    kernel_fnames = []
    for kernel in kernels:
        if not os.path.exists(os.path.join(KERNEL_DATAFOLDER, kernel)):
            download_kernel(kernel, KERNEL_DATAFOLDER)
        kernel_fnames.append(os.path.join(KERNEL_DATAFOLDER, kernel))

    return kernel_fnames


def download_kernel(kernel, KERNEL_DATAFOLDER):
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
            with tqdm.tqdm(total=total_length, ascii=True, bytes=True, desc=f'Downloading {kernel}') as pbar:
                for data in tqdm.tqdm(response.iter_content(chunk_size=4096)):
                    f.write(data)
                    pbar.update(len(data))


def get_kernels(KERNEL_DATAFOLDER, start_utc):
    if not os.path.exists(KERNEL_DATAFOLDER):
        os.mkdir(KERNEL_DATAFOLDER)

    for folder in ['ik', 'ck', 'spk', 'pck', 'fk', 'lsk', 'sclk']:
        if not os.path.exists(os.path.join(KERNEL_DATAFOLDER, folder)):
            os.mkdir(os.path.join(KERNEL_DATAFOLDER, folder))

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
            kernels.append(os.path.join('ck/', ck))
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
                kernels.append(os.path.join('ck/', ck))
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
            kernels.append(os.path.join('spk/', spk))
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
                kernels.append(os.path.join('spk/', spk))
                nspk += 1

    assert nck * nspk > 0, "Kernels not found for the given date range!"

    # load the latest updates for these
    kernels.append(os.path.join('ik/', iks[-1]))
    kernels.append(os.path.join('spk/', spks2[-1]))
    kernels.append(os.path.join('spk/', spks3[-1]))
    kernels.append(os.path.join('spk/', spks4[-1]))
    kernels.append(os.path.join('pck/', pcks[-1]))
    kernels.append(os.path.join('fk/', fks[-1]))
    kernels.append(os.path.join('sclk/', sclks[-1]))
    kernels.append(os.path.join('lsk/', lsks[-1]))
    kernels.append("spk/juno_rec_orbit.bsp")
    kernels.append("spk/juno_pred_orbit.bsp")

    return check_and_download_kernels(kernels, KERNEL_DATAFOLDER)
