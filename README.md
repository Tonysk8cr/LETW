
# LETW

LETW is an open-source initiative dedicated to developing an AI model that recognizes and understands signs from LESCO (Costa Rican Sign Language). This project provides tools and guidance for building a TensorFlow-based model that facilitates sign language interpretation and supports inclusive communication.



## Installation

!Use the code available at the main branch since the one from Dev is under development and could include errors
This project requires Python 3.8 due to Mediapipe Holistic

1. Clone the repository
```bash
git clone https://github.com/Tonysk8cr/LETW.git
```
2. Create the virtual environment
```bash
python -m venv .venv
```
3. Activate the virtual environment
```bash
# Windows:
.venv\Scripts\activate

# Linux or MacOs:
source .venv/bin/activate
```
4. Install dependencies 
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

Keep in mind that once the Test_Videos folder is automatically created, a subfolder for each action specified in the DataExtraction class will also be generated. So please make sure the list of actions is correctly defined in that class.

After that, you will need to manually place the corresponding videos into each of these action folders. The videos should follow the naming convention:
Action(1, 2, 3, 4) (e.g., Hello1.mp4, Hello2.mp4, etc.).


2. Check for incorrect path references

Since this project was originally developed and tested in a local environment, you must ensure that all file paths are correctly set for your own system.

Please review the following files and update any hardcoded paths:

App.py

DataExtraction.py

DataLabelling.py

RealtimePrediction.py

Make sure all paths point to the appropriate locations on your machine to avoid any issues during execution.

3. App.py
As mentioned earlier, App.py is the main entry point of the project.
Once executed, a menu will appear in the console, from which you can navigate and run the different functionalities of the system.




## Contributing
As mentioned at the beginning of this README, this is open source.
Please feel free to use it, and if you do, try not to charge others for it.
Keep in mind that this project was mainly developed to help those who need it most!


## Authors

- [@Tonysk8cr](https://github.com/Tonysk8cr)

