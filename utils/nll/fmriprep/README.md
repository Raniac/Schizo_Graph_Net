# NEURO-LEARN-LOCAL with fMRIPrep

## Getting Started

```bash
docker run -ti --rm \
    -v path/to/data:/data:ro \
    -v path/to/output:/out \
    poldracklab/fmriprep:<latest-version> \
    /data /out/out \
    participant
```

For further information, visit [Documentation of fMRIPrep](https://fmriprep.readthedocs.io/).

## Dataset Structure

The structure of the dataset directory should follow the standards supported in [Brain Imaging Data Structure (BIDS)](https://bids.neuroimaging.io/).

Validator for the BIDS, visit [GitHub/bids-validator](https://github.com/bids-standard/bids-validator).

```bash
docker run -ti --rm -v /path/to/data:/data:ro bids/validator /data
```