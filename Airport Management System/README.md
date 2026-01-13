[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/j0WbCUcA)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=13059269&assignment_repo_type=AssignmentRepo)
# PF-final-project
## made by:
1. Muhammad Abbas 23k-0068
2. Murtaza Jaffri 23k-0076
3. Muhammad Sabeeh 23k-0002

## This project is a airport flight management terminal. 

## project proposal:
<p>the idea was to create a program that can cater to the needs of both management staff and the people coming to the<br>
   airport. A layer of security is to be added to ensure a normal user cant access the admin panel and the data in the<br>
   file cant be manually changed. In addition to that data recycling functionality was to be added so that any and all<br>
   flight records deleted are stored in another recycled file and that data could be deleted later on. Attractive prompt<br>
   messages to be displayed to make the interface more user friendly and should have enough data manipulation options for<br>
   the management. The user panel should have all the data about flight layovers and flight timings to various destinations<br>
   which the user can access through the user panel and also should have the functionality to book flight tickets.
</p>

## data-structures used:
1. structs are used to store data about flights, tickets and dates
2. arrays are used to store strings

## It has 2 main sections:

### admin panel:
1. uses a password to login to admin panel
2. can add flights and their information to a binary file (so data cant be manually changed from the file)
3. can view a list of all available flights from a binary file (so data cant be manually changed from the file)
4. can search for specific flights using their flight number
5. can edit specific information for flights that have already been saved to the file
6. can delete a flights data from the file and store that data in a recycled file.
7. can display the list of all the tickets booked in the user panel and their data (added on sir's behest)
8. exit the program with a prompt message

### user panel:
1. can search for information on arriving flights
2. can search for information on departing flights
3. can book tickets and that data is stored in a csv file (at sir's behest)
4. A self check in system that uses a criteria for the passport numbers and ticket numbers to inform the passengers on if their flights are in business class <br> or in economy class and services offered by both are mentioned
5. passengers can choose their connecting flights and calculate how long will it take to reach their destination through these different routes
6. exit the terminal with a attractice designed prompt message.

## problems faced:
1. the search flight function and delete flight function was searching flights by country not by fligh id and this<br>
   couldve cause a issue if there were 2 flights with the same country.It was corrected by changing the program to<br>
   to search for the flight id instead.
2. The user panel didnt had a option to book flights so that was added and data was saved to a csv file by filing<br>
   and then the admin panel had to access that same file to see read that data so that functionality was added.
3. in the admin panel the option 6 was displaying the results for just a split second and then returning back most<br>
   proabably due to some loop so i added the getchar function so that the results will only be removed from the screen<br>
   when the user presses a key on the keyboard.
4. taking the program execution to different parts of the code was not working so figured out that it could be done by<br>
   labels and goto statements and implemented that.

## logic used:
<p>
   the code is basically divided into many parts.First the main function runs and asks for user input on whether to access<br>
   user panel or admin panel. If the admin panel is selected then a password will ensure secure access. Then a menu will be<br>
   showing all the functionalities and based on the user choice the relevant function for that task will be executed or on <br>
   a prompt message will be displayed. If the user panel is executed so again a menu will be displayed and this time for any<br>
   information all the data is hardcoded so using switch statements and labels all the relevant data according to the user's<br>
   choice will be displayed.Only for booking the tickets will there be a function called.On exiting an attractive prompt message<br>
   will be displayed to the user bidding them farewell.
</p>

## possible improvements:
1. data validation and verification could be added so that user and management enters data in the correct format to the files.
2. SQL could be implemented so that this code is connected to a database and SQL queries could be used to fetch data from there.
3. A graphical user interface could be implemented so that the user interface is more user friendly and attractive.
4. data encryption on the files can be implemented so that the data is ony accessible by people with authority.
5. stacks and queues could be used to store the data of flights and tickets.


# video demo <br>

## admin panel:<br>


https://github.com/NUCES-Khi/pfproject-chain_smokers/assets/75746019/0c1df525-ec0e-4fe3-8893-f815429cfbf3



## user panel: <br>


https://github.com/NUCES-Khi/pfproject-chain_smokers/assets/75746019/1bb28048-72ec-4cae-8e93-d7da69f463eb





## Contributions:

### Murtaza Jaffri:
Executed the incorporation of data in other functions, initiated calls to various functions, and designed the main function.

### Muhammad Abbas:
Coded and executed all functions as part of the implementation and implemented filing for both admin and user panel.

### Muhammad Sabeeh:
Implemented hardcoded data within the user panel and designed its interface.





   
