# Project's Name

This source code is implemented for project "Population dynamics of agent-based predator-prey system in a changing arena" in Simulation of complex adaptive sysmtems. That simulate Population dynamics for prey and predator in certain area

![]()

<!-- **Table of Contents**

- [Installation](#installation)
- [Execution / Usage](#execution--usage)
- [Technologies](#technologies)
- [Features](#features)
- [Contributing](#contributing)
- [Contributors](#contributors)
- [Author](#author)
- [Change log](#change-log)
- [License](#license) -->

## Installation
At visual studiio:
1. Create ennvironment command + shift + p 
2. Type python create environment
3. Select Venv
4. Select python version
5. Open terminal ctrl + shift +`
5. Run command `pip3 install -r requirements.txt`

On macOS and Linux:

```sh
pip3 install -r requirements.txt
```

On Windows:

```sh
pip3 install -r requirements.txt
```

## Execution / Usage

To run Group 13 project, fire up a terminal window and run the following command:
1. After install all lib complete
2. Update configuration on config.json
3. Go to project folder
4. Run main.py or `shift + F5` in visual studio

Main Library list
- numpy
- matplotlib
- pynput

## Execution by command line
MAC
```
./reindeer-project --config <CONFIG_JSON_PATH> -- result <RESULT_PATH>
```

WINDOWS - in PowerShell
```
./reindeer-project --config <CONFIG_JSON_PATH> -- result <RESULT_PATH>
```
Example
`./dist/reindeer-project --config ./config.json --result ./results/`



## Create execution file

### Pre-requisite
- Install pyinstaller by `pip3 install pyinstaller`

MAC
```sh
pyinstaller --onefile --name reindeer-project ./project/main.py
```

Windows
```sh
pyinstaller --onefile --name reindeer-project ./project/main.py
```


<!-- 
## Technologies

< Project's name > uses the following technologies and tools:

- [Python](https://www.python.org/): ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
- ...

## Features

< Project's name > currently has the following set of features:

- Support for...
- ...

## Contributing

To contribute to the development of < project's name >, follow the steps below:

1. Fork < project's name > from <https://github.com/yourusername/yourproject/fork>
2. Create your feature branch (`git checkout -b feature-new`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add some new feature'`)
5. Push to the branch (`git push origin feature-new`)
6. Create a new pull request

## Contributors

Here's the list of people who have contributed to < project's name >:

- John Doe – [@JohnDoeTwitter](https://twitter.com/< username >) – john@example.com
- Jane Doe – [@JaneDoeTwitter](https://twitter.com/< username >) – jane@example.com

The < project's name > development team really appreciates and thanks the time and effort that all these fellows have put into the project's growth and improvement.

## Author

< Author's name > – [@AuthorTwitter](https://twitter.com/< username >) – author@example.com

## Change log

- 0.0.2
    - Polish the user interface
- 0.0.1
    - First working version
- ...

## License

< project's name > is distributed under the < license > license. See [`LICENSE`](LICENSE.md) for more details. -->