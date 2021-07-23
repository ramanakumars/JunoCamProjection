# Processing and projecting JunoCam images
This is a tool to process and project JunoCam images onto a lat/lon grid. 

## Dependencies
Install the python dependencies using the `requirements.txt`:

```bash
pip3 install -r requirements.txt
```

## Installation
To run the projection code, the C extension needs to be compiled. To do this, run,
```bash
cd projection/
make clean
make
```

## Examples
See `projection.ipynb` for an example of JunoCam image projection.

To rerun the projection code, you will need to unzip the `5989-Data.zip` and `5989-ImageSet.zip`
files, which will create the `DataSet` and `ImageSet` directories. 
