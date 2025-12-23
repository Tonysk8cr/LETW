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

There are two ways to set up the project environment. The recommended approach is using `uv`.

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

### Alternative Setup with `pip`

This method is available if you prefer to manage your environment manually. However, it is not the primary supported method.

1.  **Install Python**

    Ensure you have Python 3.8 installed on your system.

2.  **Create and activate the virtual environment**
    ```bash
    python -m venv .venv
    ```
    ```bash
    # Windows:
    .venv\Scripts\activate

    # Linux or macOS:
    source .venv/bin/activate
    ```

3.  **Install dependencies**

    This will install the packages listed in `requirements.txt`. Note that this file may not be as up-to-date as the `uv.lock` file.
    ```bash
    pip install -r requirements.txt
    ```


## Usage/Examples

Once you have installed all the necessary dependencies, you should be able to run the project.
It is important to note that some changes to the code may be required, at the moment the current code is set up to avoid this, but still important to check 

This system is accessed through the App.py file, which serves as the main entry point. It manages the application's flow and allows for easier execution of the system.

1. Create the necesary folders
Once you run the App.py file (since this is a console application), you will see some output in the console. At the main menu, the first option you should choose is option 1. This will create two critical folders inside the /Test folder within your Model directory:

MP_Data: This is where the NumPy arrays will be stored later.

Test_Videos: This is where you will store your training videos.

Keep in mind that once the Test_Videos folder is automatically created, a subfolder will also be generated for each action specified in the main class under the variable called signs. Please ensure that the list of actions is correctly defined in that class.

After that, you will need to manually place the corresponding videos into each of these action folders. The videos should follow the naming convention:
Action(1, 2, 3, 4) (e.g., Hello1.mp4, Hello2.mp4, etc.).


2. Check for incorrect path references

Even though the project uses relative paths throughout its logic, it is important to verify them. Since this project was originally developed and tested in a local environment, you must ensure that all file paths are correctly set for your system.

Please review the following files and update any hardcoded paths:

App.py

DataExtraction.py

DataLabelling.py

RealtimePrediction.py

Make sure all paths point to the appropriate locations on your machine to avoid any issues during execution.

If you are using a Linux distribution, you can edit the files using:
```bash
vim <path/class.py>
```

3. App.py
As mentioned earlier, App.py is the main entry point of the project.
Once executed, a menu will appear in the console, from which you can navigate and run the different functionalities of the system.




## Contributing
As mentioned at the beginning of this README, this is open source.
Please feel free to use it, and if you do, try not to charge others for it.
Keep in mind that this project was mainly developed to help those who need it most!


## Authors

- [@Tonysk8cr](https://github.com/Tonysk8cr)

## Credits

This project is heavily based on [ActionDetectionforSignLanguage](https://github.com/nicknochnack/ActionDetectionforSignLanguage)  
created by [@nicknochnack](https://github.com/nicknochnack).  
Parts of the code were adapted and extended for specific use cases.

