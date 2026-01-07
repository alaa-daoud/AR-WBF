
## Reliability Awareness Weighted boxes fusion

Repository based on [![DOI](https://zenodo.org/badge/217881799.svg)](https://zenodo.org/badge/latestdoi/217881799)
containing Python implementation of several methods for ensembling boxes from object detection models: 

* Non-maximum Suppression (NMS)
* Soft-NMS [[1]](https://arxiv.org/abs/1704.04503)
* Non-maximum weighted (NMW) [[2]](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w14/Zhou_CAD_Scale_Invariant_ICCV_2017_paper.pdf)
* **Weighted boxes fusion (WBF)** [[3]](https://arxiv.org/abs/1910.13302) - new method which gives better results comparing to others
* **AWBF** a multi-agent system (MAS) that implement WBF in a decentralized and adaptive manner. 

In addition we added reliability awareness and adaptive iou logic that adds causally inspired layer to enhance the precision and recall

## Requirements

Python 3.*, Numpy, Numba

## Examples 
Pleas refer to main.py and AWBF_Examples folder (jupiter notebook for visualizing the bounding box evolution)



## Description of AWBF method and citation

* https://ceur-ws.org/Vol-3813/13.pdf

If you find this code useful please cite:

```
@inproceedings{daoud2024introducing,
  title={Introducing Multiagent Systems to AV Visual Perception Sub-tasks: A proof-of-concept implementation for bounding-box improvement},
  author={Daoud, Alaa and Bunel, Corentin and Gu{\'e}riau, Maxime},
  booktitle={13th International Workshop on Agents in Traffic and Transportation (ATT 2024) held in conjunction with ECAI 2024},
  year={2024},
  organization={CEUR-WS}
}
```
