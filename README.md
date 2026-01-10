# LETW

LETW is an open-source initiative dedicated to developing an AI model that recognizes and understands signs from LESCO (Costa Rican Sign Language). This project provides tools and guidance for building a TensorFlow-based model that facilitates sign language interpretation and supports inclusive communication.

For Spanish (Para Español): [![es](https://img.shields.io/badge/lang-es-yellow.svg)](https://github.com/Tonysk8cr/LETW/blob/main/README.es.md)

Tutorial:
https://youtu.be/BP-zg0obO7w



## Installation

> [!IMPORTANT]
> Use the code from the `main` branch. The `dev` branch is for development and may contain errors.

First, clone the repository:
```bash
git clone https://github.com/Tonysk8cr/LETW.git
cd LETW
```


### Recommended Setup with `uv`

This project uses `uv` for fast and reliable dependency management. It will automatically use the correct Python version defined for the project.

1.  **Install `uv`**

    If you don't have `uv` installed, please follow the installation instructions on the official [uv project page](https://astral.sh/uv/install).

2.  **Create virtual environment and sync dependencies**

    `uv` will create the virtual environment and install all the dependencies from the lock file (`uv.lock`) in a single step.
    ```bash
    uv sync
    ```

3.  **Activate the virtual environment**
    ```bash
    # Windows:
    .venv\Scripts\activate

    # Linux or macOS:
    source .venv/bin/activate
    ```
    You are now ready to proceed to the [Usage/Examples](#usageexamples) section. To run the main application, you can use `uv run ./Model/Test/App.py`.



## Usage/Examples

Once you have installed all the necessary dependencies, you should be able to run the project.
It is important to note that some changes to the code may be required, at the moment the current code is set up to avoid this, but still important to check 

This system is accessed through the App.py file, which serves as the main entry point. It manages the application's flow and allows for easier execution of the system.

In one of the most recent updates, we modified the system to support running the project across multiple backends. This change was made by decoupling the core logic from a specific TensorFlow version and aligning it with native Keras instead. TensorFlow remains the default backend.

If you wish to run the project using an alternative backend, you must start the application with the appropriate parameters, as shown below:

JAX backend:
```bash
uv run --group jax env KERAS_BACKEND=jax python ./Model/Test/App.py
```

PyTorch backend:
```bash
uv run --group torch env KERAS_BACKEND=torch python ./Model/Test/App.py
```


1. Create the necesary folders
Once you run the App.py file (since this is a console application), you will see some output in the console. At the main menu, the first option you should choose is option 1. This will create two critical folders inside the /Test folder within your Model directory:

MP_Data: This is where the NumPy arrays will be stored later.

Test_Videos: This is where you will store your training videos.

Keep in mind that once the Test_Videos folder is automatically created, a subfolder will also be generated for each action specified in the main class under the variable called signs. Please ensure that the list of actions is correctly defined in that class.

After that, you will need to manually place the corresponding videos into each of these action folders. The videos should follow the naming convention:
Action(1, 2, 3, 4) (e.g., Hello1.mp4, Hello2.mp4, etc.).







## Contributing
As mentioned at the beginning of this README, this is open source.
Please feel free to use it, and if you do, try not to charge others for it.
Keep in mind that this project was mainly developed to help those who need it most!


## Developed by

- [@Tonysk8cr](https://github.com/Tonysk8cr)
- [@eariassoto](https://github.com/eariassoto)

## Credits

This project is heavily based on [ActionDetectionforSignLanguage](https://github.com/nicknochnack/ActionDetectionforSignLanguage)  
created by [@nicknochnack](https://github.com/nicknochnack).  
Parts of the code were adapted and extended for specific use cases.

