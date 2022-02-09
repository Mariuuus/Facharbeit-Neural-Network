# Facharbeit Neural Network
## ON WINDOWS
#### Setting up for Usage
> installing independences out of pip (shell)

install [python 3](https://www.python.org/ftp/python/3.10.2/python-3.10.2-amd64.exe)

>create the virtual enviroment in powershell and install the independences via pip and the requirement.txt
```
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```
#### Running the Neural Network
>just execute the file **final_programm.py** via:
>make sure that youre in the right directory via cd *path*
```
.\.venv\Scripts\activate
python .\final_programm.py
```
>programm is now started and should be self explaining


## ON LINUX
#### Setting up for Usage
> installing independences out of pip (shell)
```
sudo apt install python3.8
sudo apt install python3-pip
sudo apt install python3.8-venv
sudo apt-get install python-tk
```
>create the virtual enviroment and install the independences via pip and the requirement.txt
```
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```
#### Running the Neural Network
>just execute the file **final_programm.py** via:
>make sure that youre in the right directory via cd *path*
```
source env/bin/activate
python3 final_programm.py
```
>programm is now started and should be self explaining
