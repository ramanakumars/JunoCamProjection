'''
    Download the ImageSet.zip and DataSet.zip from the missionjuno website
    for a given perijove pass for JunoCam to the current folder.

    usage: get_junocam_data.py [-h] PJ

    Download JunoCam data from the missionjuno website

    positional arguments:
      PJ          Number of the perijove pass

    optional arguments:
      -h, --help  show this help message and exit


    Ex. python3 get_junocam_data.py 12 will download perijove 12 images
    into the PJ12/ folder where the script is called from.
'''

import re
import subprocess
import requests
from bs4 import BeautifulSoup
import argparse

base_url = 'https://www.missionjuno.swri.edu'


def get_tag_from_text(line, keyword, tag='a', contains={}):
    '''
        Retrieve a keyword from a HTML tag that matches
        a format check

        Inputs
        ------
        line : string
            the full line containing multiple HTML tags
        keyword : string
            the keyword in the HTML tag that we want
            the value for (e.g., class, href)
        tag : string
            the type of tag (e.g., a for hyperlinks, etc.)
        contains : dict
            dictionary containing the keyword and value for
            the tag in question (e.g., retrieve only tags that
            have class="nav". In that case contains={'class': 'nav'}.

        Outputs
        _______
        out : list
            list of dictionaries with the tag name and the keyword
            value that was queried, e.g.,
            out = {'tag': 'a', 'href': 'www.google.com'}

    '''
    # parse the line into separate tags
    tag_info = parse_html_from_line(line)

    out = []
    if tag_info is not None:
        # loop through each tag and check
        # if it's the one we're looking for
        for tagi in tag_info:
            if tagi['tag'] == tag:
                check = 0
                # validate against all keys
                # in the check
                for kw in contains.keys():
                    # make sure the tags contain the
                    # key that we're matching against
                    if kw in tagi.keys():
                        # and then make sure that the
                        # value of the key matches (discounting
                        # whitespace)
                        if (tagi[kw].strip() == contains[kw].strip()):
                            check += 1
                if check == len(contains.keys()):
                    out.append(tagi[keyword])

    return out


def parse_html_from_line(line):
    '''
        Parse multiple HTML tags
        from a single line

        Inputs:
        -------
        line : string
            the line containing HTML tags

        Outputs:
        --------
        out : list
            list of dictionaries containing the
            tag along with its keywords, e.g.,
            <a href='www.google.com' class='google'>
            will be parsed into
            [{'tag': 'a', 'href': 'www.google.com',
            'class': 'google}]
    '''

    # clean up
    line = line.strip()
    line = line.replace('\t', '')

    tag_pattern = r'<(\w+) (.*?)>'
    kw_pattern = r'(\w+)=\"(.*?)\"'

    tag_match = re.findall(tag_pattern, line)

    out = []
    for matchi in tag_match:
        keyi = {}
        keyi['tag'] = matchi[0]

        keywords = re.findall(kw_pattern, matchi[1])

        for keyword in keywords:
            keyi[keyword[0]] = keyword[1]
        out.append(keyi)
    return out


# the perijove thumbnail link has this class
PJ_search = {'class': "modal_partial_trigger modal_gallery_item".split(' ')}
# the metadata/image download link has this class
zip_search = {'class': "marR download_zip".split(' ')}

heading_pattern = r'<h1 class=\"fake_h2 padB\">(.*?)<\/h1>'


def get_data_perijove(PJ):
    # get the URL for that perijove
    PJ_url = base_url + "/junocam/processing?source=junocam&phases[]=PERIJOVE+%d&perpage=72" % PJ

    # with urllib.request.urlopen(PJ_url) as response:
    #     responseText = response.read().decode()
    #
    #     lines = responseText.split('\n')
    #
    #     # get a list of IDs that correspond to the
    #     # junocam mosaics
    #     IDs = []
    #     for line in lines:
    #         idi = get_tag_from_text(line, 'href', contains=PJ_search)
    with requests.get(PJ_url) as response:
        soup = BeautifulSoup(response.text, 'html.parser')
    # IDs = [a['href'] for a in soup.find_all('a') if a.get('class', None) == PJ_search['class']]
    IDs = [a.get('href') for a in soup.find_all('a') if all([item in a.get('class', []) for item in PJ_search['class']])]

    print("PJ %d: Found %d IDs" % (PJ, len(IDs)))
    zips = []
    # with the IDs now parsed, loop through each
    # ID, open the page for that ID and query the
    # metadata and imageset link
    for ID in IDs:
        sub_url = base_url + ID
        print(sub_url)
        with requests.get(sub_url) as response:
            soup = BeautifulSoup(response.text, 'html.parser')

        header = soup.find(lambda tag: tag.name == 'h1' and tag.get('class', []) == ['fake_h2', 'padB']).contents[0]

        try:
            # metadata = soup.find(lambda tag: tag.get('id', None) == 'metadataDropdown').contents
            metadata = soup.find('dl')
            filters = metadata.div.dd.contents[0]
        except AttributeError:
            continue

        print(header, filters)

        # ignore radiation trend monitoring
        if 'Radiation' in header:
            continue

        # we don't want the methane band
        if ("BLUE, GREEN, RED" in filters):
            zipi = soup.find_all(lambda tag: tag.name == 'a' and tag.get('class', []) == zip_search['class'])

            for zipii in zipi:
                if zipii not in zips:
                    zips.append(zipii['href'])
                    # download the zip into the respective PJ folder
                    subprocess.run(["wget", "-NS", "-nv", "--content-disposition", "-P", "PJ%d/" % PJ, base_url + zipii['href']])

            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download JunoCam data from the missionjuno website')

    parser.add_argument('PJ', metavar='PJ', type=int, nargs=1, help='Number of the perijove pass')

    args = parser.parse_args()

    if len(args.PJ) == 1:
        get_data_perijove(args.PJ[0])
