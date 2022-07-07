# smg-mvdepthnet

This Python package provides a wrapper for MVDepthNet.

It is a submodule of [smglib](https://github.com/sgolodetz/smglib), the open-source Python framework associated with our drone research in the [Cyber-Physical Systems](https://www.cs.ox.ac.uk/activities/cyberphysical/) group at the University of Oxford.

### Development Installation (as part of smglib)

Note: Please read the [top-level README](https://github.com/sgolodetz/smglib/blob/master/README.md) for smglib before following these instructions.

1. Open the terminal.

2. Activate the Conda environment, e.g. `conda activate smglib`.

3. If you haven't already installed PyTorch, install it now. In our case, we did this via:

   ```
   pip install https://download.pytorch.org/whl/cu111/torch-1.9.1%2Bcu111-cp37-cp37m-win_amd64.whl
   pip install https://download.pytorch.org/whl/torchaudio-0.9.1-cp37-cp37m-win_amd64.whl
   pip install https://download.pytorch.org/whl/cu111/torchvision-0.10.1%2Bcu111-cp37-cp37m-win_amd64.whl
   ```

   However, you may need a different version of PyTorch for your system, so change this as needed. (In particular, the latest version will generally be ok.)

4. Change to the `<root>/smg-mvdepthnet/smg/external/mvdepthnet` directory.

5. Check out the `smg-mvdepthnet` branch.

6. Change to the `<root>/smg-mvdepthnet` directory.

7. Check out the `master` branch.

8. Run `pip install -e .` at the terminal.

9. Download the pre-built MVDepthNet model (`opensource_model.pth.tar`) and set (at a system level, not in the terminal) an environment variable called `SMGLIB_MVDEPTH_MODEL_PATH` that points to its location on disk.

   (Note: It's currently unavailable from the original download site, so please contact us if you need a copy.)

### Publications

If you build on this framework for your research, please cite the following paper:
```
@inproceedings{Golodetz2022TR,
author = {Stuart Golodetz and Madhu Vankadari* and Aluna Everitt* and Sangyun Shin* and Andrew Markham and Niki Trigoni},
title = {{Real-Time Hybrid Mapping of Populated Indoor Scenes using a Low-Cost Monocular UAV}},
booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
month = {October},
year = {2022}
}
```

### Acknowledgements

This work was supported by Amazon Web Services via the [Oxford-Singapore Human-Machine Collaboration Programme](https://www.mpls.ox.ac.uk/innovation-and-business-partnerships/human-machine-collaboration/human-machine-collaboration-programme-oxford-research-pillar), and by UKRI as part of the [ACE-OPS](https://gtr.ukri.org/projects?ref=EP%2FS030832%2F1) grant. We would also like to thank [Graham Taylor](https://www.biology.ox.ac.uk/people/professor-graham-taylor) for the use of the Wytham Flight Lab, [Philip Torr](https://eng.ox.ac.uk/people/philip-torr/) for the use of an Asus ZenFone AR, and [Tommaso Cavallari](https://uk.linkedin.com/in/tcavallari) for implementing TangoCapture.
