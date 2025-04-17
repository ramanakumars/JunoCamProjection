# Processing and projecting JunoCam images

[<img src="https://readthedocs.org/projects/junocamprojection/badge/?version=latest&style=flat-default">](https://junocamprojection.readthedocs.io/en/latest/)

This is a tool to process and project JunoCam images onto a lat/lon grid.

>[!NOTE]
>The C extension will only run on Linux amd64

## Dependencies
Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install the python dependencies using the `uv sync`:
```bash
uv sync
```

## Installation
To run the projection code, the C extension needs to be compiled. To do this, run,
```bash
cd junocam_projection/
make clean
make
```

Then, install the package using a Python package manager. For example:

```bash
pip3 install .
```

This will install the `junocam_projection` package into your Python environment. 

## Examples

To run the jupyter notebook:
```bash
uv run --with jupyter jupyter lab
```

### Projecting a single image 

See `examples/projection.ipynb` for an example of JunoCam image projection.

To rerun the projection code, you will need to unzip `8724-Data.zip` and `8724-ImageSet.zip`
in the `examples/` folder , which will create the `DataSet` and `ImageSet` directories. 

### Multi-image mosaicing
See `examples/mosaic.ipynb` for an example of mosaicing two images. To rerun the code, 
you will need to unzip all the zip files in the `examples/` folder. 

Example of mosaic from multiple images from perijove 27:
![PJ27](https://raw.githubusercontent.com/ramanakumars/JunoCamProjection/master/examples/PJ27_mosaic_RGB.png)
