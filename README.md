# F-TöRF: Flowed Time-of-Flight Radiance Fields

### [Project Page](https://visual.cs.brown.edu/projects/ftorf-webpage/) | [Paper](https://mmehas.github.io/files/ftorf.pdf) | [F-TöRF Data](https://1drv.ms/f/c/4dd35d8ee847a247/EsiF6mb15ZlKlTZmg8N_OIcBCaQGUmWWVNOldMTaRsQXeQ?e=eIy7Rz)
[Mikhail Okunev](https://mmehas.github.io),
[Marc Mapeke](https://www.linkedin.com/in/marcmapeke/),
[Benjamin Attal*](https://benattal.github.io/),
[Christian Richardt](https://richardt.name/)<sup>∞</sup>,
[Matthew O'Toole](https://www.cs.cmu.edu/~motoole2)<sup>*</sup>,
[James Tompkin](https://www.jamestompkin.com) <br>
Brown University, <sup>∞</sup>Meta Reality Labs, *Carnegie Mellon University

## Environment setup
To set up the environment, run<br>
```
./install_env.sh
```

The script should install a conda environment ```ftorf``` and activate it.

This setup was tested on NVIDIA 3090 RTX and CUDA 11.8, version adjustments for tensorflow version might be required for other configurations.

## Obtaining the data

Download the [data](https://1drv.ms/f/c/4dd35d8ee847a247/EsiF6mb15ZlKlTZmg8N_OIcBCaQGUmWWVNOldMTaRsQXeQ?e=eIy7Rz) and save it to the ```data/``` folder in the repo directory. Then run ```prepare_data.py``` to unpack the data.

## Training the models from scratch

To train on one of the synthetic scenes (```sliding_cube, occlusion, speed_test_texture, speed_test_chair, arcing_cube, z_motion_speed_test, acute_z_speed_test```) run

```
./train_synthetic.sh <scene name>
```

For real scenes (```pillow, baseball, fan, target1, jacks1```) run

```
./train_real.sh <scene name>
```

The training is going to take 3-5 days. You will find the logs under ```logs/<scene name>```.
In case your GPU does not have enough memory, try to reduce the batch size using ```N_rand``` parameter (default is 256).

## Rendering

If you only want to render the videos using your trained model, run

```
./train_synthetic.sh <scene name> --render_only
```
or
```
./train_real.sh <scene name> --render_only
```

## Citation

```
@inproceedings{okunev2024flowed,
    title={Flowed Time of Flight Radiance Fields},
    author={Okunev, Mikhail and Mapeke, Marc and Attal, Benjamin and Richardt, Christian and O’Toole, Matthew and Tompkin, James},
    year={2024},
    organization={European Conference on Computer Vision}
}
```
