# Measuring cognitive effort using tabular transformer-based language models of electronic health record-based audit log action sequences
[![JAMIA](https://img.shields.io/badge/JAMIA-ocae171-b31b1b.svg)](https://doi.org/10.1093/jamia/ocae171) [![License](https://img.shields.io/badge/License-AGPLv3-darkgreen.svg)](https://opensource.org/licenses/agpl-v3)

This repo contains code for training and evaluating transformer-based tabular language models for Epic EHR audit logs. Each branch contains a different variation of our models, the principal one being the `app-attending` branch, which contains code for our models trained specifically for advanced practice providers (APPs) and attending physicians.

## Installation
Use `pip install -r requirements.txt` to install the required packages. If updated ` pipreqs . --savepath
requirements.txt --ignore Sophia` to update. Use `git submodule update --init --recursive` to get Sophia for training.

This project uses pre-commit hooks for `black` if you would like to contribute. To install run `pre-commit install`.

To use our models for cross-entropy loss, see `entropy.py` for a broad overview of the setup needed. Since they're built with `transformers` you can also use these models for generative tasks in nearly the same way as any other language model. See `gen.py` for an example of how to do this.

## Citation

Please cite our paper if you use this code in your own work:

```
 @article{Kim_Warner_Lew_Lou_Kannampallil_2024,
      title={Measuring cognitive effort using tabular transformer-based language models of electronic health record-based audit log action sequences},
      ISSN={1527-974X},
      DOI={10.1093/jamia/ocae171},
      author={Benjamin C. Warner and Thomas Kannampallil and Seunghwan Kim},
      journal={Journal of the American Medical Informatics Association},
      author={Kim, Seunghwan and Warner, Benjamin C and Lew, Daphne and Lou, Sunny S and Kannampallil, Thomas},
      year={2024},
      month=jul,
      pages={ocae171} 
}

```

