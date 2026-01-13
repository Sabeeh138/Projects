/*Airport terminal system*/



#include <stdio.h>
#include <string.h>
#include <conio.h>
#include <stdlib.h>
// stucutre for the date
struct date
{
	int d;
	int m;
	int y;
};

int abc;

// structure for flight info
struct flight
{
	int flightnum;
	char country[30];
	int timing;
	struct date flightdate;
} flightinfo;

//structure to hold ticket data 
struct ticket 
{
	int ticketnum;
	char destination[30];
	char owner[50];
	int age;
	int day;
	int month;
	int year;
}

// designed message to be displayed on start of the program
headMessage(const char *title)
{
	system("cls");
	printf("\t\t\t###########################################################################");
	printf("\n\t\t\t############                                                   ############");
	printf("\n\t\t\t############            Flight management System              ############");
	printf("\n\t\t\t############                                                   ############");
	printf("\n\t\t\t###########################################################################");
	printf("\n\t\t\t---------------------------------------------------------------------------\n");
	printf("\t\t\t\t\t\t\t%s", title);
	printf("\n\t\t\t----------------------------------------------------------------------------");
}

// another designed message to be displayed on the start of the program
start()
{
	headMessage("PF PROJECT");
	printf("\n\n\n\n\n");
	printf("\n\t\t\t  ********************************************************\n");
	printf("\n\t\t\t        =============================================");
	printf("\n\t\t\t        =                 WELCOME                   =");
	printf("\n\t\t\t        =                   TO                      =");
	printf("\n\t\t\t        =                 FLIGHT                    =");
	printf("\n\t\t\t        =               MANAGEMENT                  =");
	printf("\n\t\t\t        =                 SYSTEM                    =");
	printf("\n\t\t\t        =============================================");
	printf("\n\n\t\t\t  ********************************************************\n");
	printf("\n\n\n\t\t\t Press Enter to continue.....");
	getch(); // this will pause the program until the user has pressed some key on the keyboard
}

// adds a new flight record  to the database
add()
{
	FILE *fp;
	fp = fopen("lms.bin", "ab+"); // opens a binary file
	// error handling
	if (fp == NULL)
	{
		printf("\n\t\t\tFile is not opened\n");
		exit(1);
	}
	headMessage("ADD flightS");
	printf("\n\n\t\t\tENTER flight DETAILS BELOW:");
	printf("\n\t\t\t---------------------------------------------------------------------------\n");

	// entering data about the flight
	printf("\n\t\t\tflight number: ");
	scanf("%d", &flightinfo.flightnum);
	fflush(stdin); // clearing the buffer

	printf("\n\t\t\tflight country: ");
	gets(flightinfo.country);
	fflush(stdin);

	printf("\n\t\t\tTiming: ");
	scanf("%d", &flightinfo.timing);
	fflush(stdin);

	printf("\n\t\t\tEnter date(dd): ");
	scanf("%d", &flightinfo.flightdate.d);
	fflush(stdin);

	printf("\n\t\t\tEnter month(mm): ");
	scanf("%d", &flightinfo.flightdate.m);
	fflush(stdin);

	printf("\n\t\t\tEnter year(yyyy): ");
	scanf("%d", &flightinfo.flightdate.y);
	fflush(stdin);

	// writes the whole flight structure to the file
	fwrite(&flightinfo, sizeof(flightinfo), 1, fp);
	fclose(fp);
}

// id country error corrected for this function
search() // function to search for data of specific flights by typing in the country
{
	int found = 1;
	int searchflight;
	FILE *fp;
	fp = fopen("lms.bin", "rb"); // opening the binary file

	// error handling
	if (fp == NULL)
	{
		printf("\n\t\t\tFile is not opened\n");
		exit(1);
	}
	headMessage("SEARCH flightS");

	printf("\n\n\t\t\tEnter flight ID to search: ");
	scanf("%d", &searchflight);
	fflush(stdin); // clearing the input buffer in case input of wrong data type is made

	rewind(fp); // function to bring back program execution to the start of the binary file
	while (fread(&flightinfo, sizeof(flightinfo), 1, fp) == 1)
	{
		if (flightinfo.flightnum == searchflight) // compares user input to the data stored
		{
			// if its a match then all the relevant data is printed
			printf("\n\t\t\tflight ID = %d\n", flightinfo.flightnum);
			printf("\t\t\tflight country = %s\n", flightinfo.country);
			printf("\t\t\t Timings = %d\n", flightinfo.timing);
			printf("\t\t\tIssue date(day/month/year) =  %d/%d/%d\n", flightinfo.flightdate.d,
				   flightinfo.flightdate.m, flightinfo.flightdate.y);
			found = 0;
			break;
		}
	}
	if (found == 1)
	{
		printf("\n\t\t\tNo Record");
	}
	fclose(fp);
	printf("\n\n\n\t\t\tPress Enter to go to main menu.....");
	getchar();
}

view() // will display all the flghts available and their data
{
	int found = 0;
	int count = 1;
	headMessage("flightS DETAILS");
	FILE *fp = fopen("lms.bin", "rb");
	// error handling
	if (fp == NULL)
	{
		printf("\n\t\t\tFile is not opened\n");
		exit(1);
	}
	rewind(fp);												   // takes the prompt back to the start of the file
	while (fread(&flightinfo, sizeof(flightinfo), 1, fp) == 1) // condition that will keep executing till their is data avalable in the fil or till EOF
	{
		printf("\n\t\t\tflight %d\n\n", count);
		printf("\t\t\tflight number = %d\n", flightinfo.flightnum);
		printf("\t\t\t Country= %s\n", flightinfo.country);
		printf("\t\t\tTimings = %d\n", flightinfo.timing);
		printf("\t\t\tIssue date(day/month/year) =  (%d/%d/%d)\n\n", flightinfo.flightdate.d,
			   flightinfo.flightdate.m, flightinfo.flightdate.y);
		found = 1;
		count++;
	}
	fclose(fp);
	if (found == 0)
	{
		printf("\n\t\t\tNo Record");
	}
	printf("\n\n\t\t\tPress any key to go to main menu.....");
	fflush(stdin);
	getchar();
}

removeflight() // function to remove a flight record
{
	int found = 0;
	int deleteid = 0; // id of the flight record we want to delete
	char country[30] = {0};
	FILE *fp = NULL;
	FILE *fp1 = NULL;
	int status = 0;
	headMessage("Delete flights Details");
	fp = fopen("lms.bin", "rb");
	// error handling
	if (fp == NULL)
	{
		printf("File is not opened\n");
		exit(1);
	}

	fp1 = fopen("recycle.bin", "wb"); // opening another file to store the deleted records in case of need again
	// error handling
	if (fp1 == NULL)
	{
		fclose(fp);
		printf("File is not opened\n");
		exit(1);
	}

	printf("\n\t\t\tEnter flight ID NO. for delete:");
	scanf("%d", &deleteid);
	fflush(stdin);

	while (fread(&flightinfo, sizeof(flightinfo), 1, fp))
	{
		if (flightinfo.flightnum == deleteid)
		{
			found = 1;
		}
		else
		{
			fwrite(&flightinfo, sizeof(flightinfo), 1, fp1); // adding the deleted record to the recycling file
		}
	}
	if (found == 1)
	{
		printf("\n\t\t\tRecord deleted successfully.....");
	}
	else
	{
		printf("\n\t\t\tNo Record Found.....");
	}
	fclose(fp);
	fclose(fp1);
	remove("lms.bin");				  // deleting the original file
	rename("recycle.bin", "lms.bin"); // renaming the file with the deleted records with the name of the original file
	printf("\n\n\t\t\tPress any key to go to main menu.....");
	getchar();
}

// id country error corrected in this function as well
edit() // to edit a flight record
{
	int found = 1;
	int editflight; // id number of the file you want to edit
	FILE *fp;
	fp = fopen("lms.bin", "rb+");
	// error handling
	if (fp == NULL)
	{
		printf("\n\t\t\tFile is not opened\n");
		exit(1);
	}
	headMessage("EDIT flightS");

	printf("\n\n\t\t\tEnter flight Country to edit: ");
	scanf("%d", &editflight);
	fflush(stdin);

	rewind(fp);
	while (fread(&flightinfo, sizeof(flightinfo), 1, fp) == 1)
	{

		if (flightinfo.flightnum == editflight)
		{

			// asking for the input of the record again
			printf("\t\t\tenter the new data you want\n");
			printf("\t\t\tPrevious flight number = %d\n", flightinfo.flightnum);
			printf("\t\t\tPrevious flight country = %s\n", flightinfo.country);
			printf("\t\t\tPrevious flight timings = %d\n", flightinfo.timing);
			printf("\t\t\tPrevious Issue date(day/month/year) =  (%d/%d/%d)\n\n", flightinfo.flightdate.d,
				   flightinfo.flightdate.m, flightinfo.flightdate.y);

			printf("\n\t\t\tNew flight number: ");
			scanf("%d", &flightinfo.flightnum);
			fflush(stdin);
			printf("\t\t\tNew flight country: ");
			gets(flightinfo.country);
			fflush(stdin);
			printf("\t\t\tNew timings: ");
			scanf("%d", &flightinfo.timing);
			fflush(stdin);
			printf("\t\t\tNew date(dd): ");
			scanf("%d", &flightinfo.flightdate.d);
			fflush(stdin);
			printf("\t\t\tNew month(mm): ");
			scanf("%d", &flightinfo.flightdate.m);
			fflush(stdin);
			printf("\t\t\tNew year(yyyy): ");
			scanf("%d", &flightinfo.flightdate.y);
			fflush(stdin);
			fseek(fp, sizeof(flightinfo), SEEK_CUR);
			fwrite(&flightinfo, sizeof(flightinfo), 1, fp);
			found = 0;
			break;
		}
	}

	if (found == 1)
	{
		printf("\n\t\t\tNo Such Record");
	}
	else
	{
		printf("Record Edited Successfully.....");
	}
	fclose(fp);
	printf("\n\n\n\t\t\tPress Enter to go to main menu.....");
	getchar();
}

//function to store ticket data in a csv file
void bookticket() {
	//struct variable to store the value
	struct ticket t;
	
	//asking the user for the data
	printf("enter the ticket number:");
	scanf("%d",&t.ticketnum);
	printf("\n");
	fflush(stdin);
	
	printf("enter the destination:");
	gets(t.destination);
	printf("\n");
	fflush(stdin);
	
	printf("enter the buyers name:");
	gets(t.owner);
	printf("\n");
	fflush(stdin);
	
	printf("enter the buyers age:");
	scanf("%d",&t.age);
	printf("\n");
	fflush(stdin);
	
	printf("enter the day of purchase(dd):");
	scanf("%d",&t.day);
	printf("\n");
	fflush(stdin);
	
	printf("enter the month of purchase(mm):");
	scanf("%d",&t.month);
	printf("\n");
	fflush(stdin);
	
	printf("enter the year of purchase(yyyy):");
	scanf("%d",&t.year);
	printf("\n");
	fflush(stdin);
	
	FILE *file = fopen("tickets.csv", "a"); //csv flle to store user ticket data
	//error handling
	if (file == NULL) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }
    
    //writing the data to the file
    fprintf(file, "%d,%s,%s,%d,%d,%d,%d\n", t.ticketnum, t.destination, t.owner,t.age,t.day,t.month,t.year);
    fclose(file);
    printf("ticket booked successfully\n");
}

//simple function with a pointer to a struct as a argument
//this function displays the ticket data
void printTicket(struct ticket *ptrstruct) {
    printf("\t\t\tTicket Number: %d\n", ptrstruct->ticketnum);
    printf("\t\t\tDestination: %s\n", ptrstruct->destination);
    printf("\t\t\tOwner: %s\n", ptrstruct->owner);
    printf("\t\t\tAge: %d\n", ptrstruct->age);
    printf("\t\t\tDate: %d/%d/%d\n", ptrstruct->day, ptrstruct->month, ptrstruct->year);
    printf("\n");
    getchar();
    
}

//function to display tickets in admin panel from a file created in the user panel
int displaytickets() {
	//opening file in read mode
	FILE *file = fopen("tickets.csv", "r");
    //error handling
    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }
    
    //pointer to struct to the ticket struct
    struct ticket t2;
    struct ticket *ptrstruct=&t2;
    
    printf("\t\t\tkeep pressing enter to keep generating more ticket info:\n\n\n");
    //fscanf function checks how many data items are there and kep running the loop till their is data
    while (fscanf(file, "%d,%49[^,],%49[^,],%d,%d,%d,%d\n", &t2.ticketnum, t2.destination, t2.owner,&t2.age,&t2.day,&t2.month,&t2.year) == 7) {
        printTicket(ptrstruct);
    }

    fclose(file);
    
}

menu()
{ // admin panel menu
	int choice = 0;
	do
	{
		headMessage("MAIN MENU");
		printf("\n\n\n\t\t\t1.Add flights");
		printf("\n\t\t\t2.Search flights");
		printf("\n\t\t\t3.View flight List");
		printf("\n\t\t\t4.Edit flight");
		printf("\n\t\t\t5.Delete flight");
		printf("\n\t\t\t6.display all the ticket booked yet");
		printf("\n\t\t\t0.Exit");
		printf("\n\n\n\t\t\tEnter choice => ");
		scanf("%d", &choice);
		
		switch (choice) // based on user option choice calling the relevant function
		{
		case 1:
			add();
			break;
		case 2:
			search();
			break;
		case 3:
			view();
			break;
		case 4:
			edit();
			break;
		case 5:
			removeflight();
			break;
		case 6:
			system("cls");
			displaytickets();
		    break;
				
		case 0:
			printf("\n\n\n\t\t\t\tThank you!\n\n\n\n\n");
			exit(1);
			break;
		
		default:
			printf("\n\n\n\t\t\tINVALID INPUT! Try again...");
		}
	} while (choice != 0);
}

//function to print a exiting message
void goodbye() {
	printf("\n\n\n\n\n\n\n\n\n");
	printf("\t\t\t                                  xxx\n");
	printf("\t\t\t                                xxxxxxx\n");
	printf("\t\t\t                            xxxxxxxxxxxxxxx\n");
	printf("\t\t\t                        xxxxxxxxxxxxxxxxxxxxxxx\n");
	printf("\t\t\t                    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
	printf("\t\t\t                xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
	printf("\t\t\t            xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
	printf("\t\t\t        xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
	printf("\t\t\t    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
	printf("\t\t\txxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
	printf("\t\t\t                 THANK YOU FOR CHOOSING HEATHROW AIRPORT\n");
	printf("\t\t\t                        Please come again soon\n");
	printf("\t\t\txxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
	printf("\t\t\t    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
	printf("\t\t\t        xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
	printf("\t\t\t            xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
	printf("\t\t\t                xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
	printf("\t\t\t                    xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
	printf("\t\t\t                        xxxxxxxxxxxxxxxxxxxxxxx\n");
	printf("\t\t\t                            xxxxxxxxxxxxxxx\n");
	printf("\t\t\t                                xxxxxxx\n");
	printf("\t\t\t                                  xxx\n");
}



//labels and goto statements have been used in this code to control program flow
int main()
{
	int j, k, fln, select, tn, b;
	int com, i, count = 1, choice, select2, a[256], z;
	char ch, psn[8], menreturn, rpass[11] = "project123" /*the password to login in the admin panel*/, id[30], pass[30], c,key;

	printf("Welcome to Heathrow airports flight management system\n\n\n\n\nEnter 1 for Admin pannel and 2 for User terminal ");
	scanf("%d", &z);
	
	switch (z)
	{
	case 1://admin panel opening
	password:
		printf("Enter Password:\n");
		fflush(stdin);
		gets(pass);				   // password is project123
		com = strcmp(pass, rpass); // comparing the user entered password to the actual password
		if (com != 0)
		{
			printf("\n\n\nIncorrect password!\n\n");
			goto password; // when incorrect password entered go to label password
		}
		start();
		menu();
		break;

	case 2: //user panel

	menu:
		system("cls"); //clears the console screen
		printf("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
		printf("               **Welcome to Heathrow Airport**\n");
		printf("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n");
		printf("\n1. Arrival Information");
		printf("\n2. Departure Information");
		printf("\n3. Self check-in");
		printf("\n4. Plan a connection");
		printf("\n5. book a flight"); //changed
		printf("\n6. EXIT PROGRAM\n\n");
		printf("<<ENTER CORRESPONDING NUMBER>>\n");
		int choice;
		scanf("%d", &choice);
		system("cls");

		switch (choice) 
		{
		default:
			printf("\nOption currently unavailable!");
			goto menu; // when invalid input is taken
			break;

		case 6: //option for exiting the program
		    goodbye();
			return 0;
			break;
			
		case 5: //extar part added on sir's request
			bookticket();
			printf("press 'y' to return to menu or press 'n' to exit:\n");
			key = getch();//will store the key pressed
			if (key=='y') {
				goto menu; //take the program back to menu
			} else {
				return 0; //exit from the program
			}
			break;

		case 3: //self checkin options 
			printf("Enter your passport number");
			scanf("%s", &psn);
			if (strlen(psn) != 8) //password validation
			{
				printf("\n passport number should have 8 digits.\nInvalid input");
			}
			else //if password is validated then ticket number is checked
			{
				printf("\nEnter ticket number");
				scanf("%d", &tn);
				if (tn % 10 == 0) //ticket number validation
				{
					printf("You are clear for boarding");
//					for (i = 0; i < 256; i++)
//					{
//						for (j = 0; j < 256; j++)
//						{
//							a[i] = j;
//						}
//					}
//
//					for (; k < 256; k++)
//					{
//						printf("Your seat number is %d", a[k]);
//						break;
//					} // faulty code

					printf("\nYou are a economy class holder");
					printf("\nYou could wait in the CBA lounge");
					printf("\nFor a upgrade please refer to the respective airline manager at the front desk");
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoicea6: //label 
					scanf("%c", &menreturn);
					if (menreturn == 'y')
					{
						system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoicea6; //if input of characters is wrong then the code execution will return to the label name point
					}
					break;
				}
				else if (tn % 10 == 1 || tn % 10 == 2) //business class ticket validation
				{
					printf("You are clear for boarding\n");
//					for (i = 0; i < 256; i++)
//					{
//						for (j = 0; j < 256; j++)
//						{
//							a[i] = j;
//						}
//					}
//
//					for (; k < 256; k--)
//					{
//						printf("Your seat number is %d\n", a[k]);
//						break;
//					}

					printf("\nYou are a Buisness/First class holder");
					printf("\nYou could wait in the Air lounge and check out the refreshments and spa");
					printf("\nThe lounge includes: \nA buffet \nA spa \nWine tasting \nA cinema \n A infinity pool");
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoicea80: //label for code to return at some other point in the code
					scanf("%c", &menreturn);
					if (menreturn == 'y')
					{
						system("cls"); //clears the command console
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoicea80; //keep returning code execution to label until correct input is added
					}
					break;
				}
				
			// hardcoding all the flight data of different cities
			case 1: //all hardcoded flight data
				printf("Enter number to search for Flight(1-10)=\n");
				printf("\n1-Belfast");
				printf("\n2-Lisbon");
				printf("\n3-Vienna");
				printf("\n4-Zurich");
				printf("\n5-Barcelona");
				printf("\n6-Madrid");
				printf("\n7-Paris");
				printf("\n8-Glasgow");
				printf("\n9-Amsterdam");
				printf("\n10-Warsaw\n");
				scanf("%d", &fln);
				switch (fln)
				{
				case 1:
					system("cls");
					printf("\n Belfast");
					printf("\n BA1414");
					printf("\nBritish Airway\n");
					printf("\nTERMINAL 5");
					printf("\n Departure time 06:00");
					printf("\n Boarding time  03:45");
					printf("\n Friday 23 December 2022");
					printf("\n Expected arrival in Belfast 7:25");
					printf("\n Status: On time");
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoicea1:
					scanf("%c", &menreturn);
					if (menreturn == 'y')
					{
						+system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoicea1;
					}
					break;

				case 2:
					system("cls");
					printf("\n Lisbon");
					printf("\nTap Portugal\n");
					printf("\nTERMINAL 2");
					printf("\n TP 1363");
					printf("\n Departure time 06:00");
					printf("\n Boarding time  03:45");
					printf("\n Friday 23 December 2022");
					printf("\n Expected arrival in Lisbon 8:45");
					printf("\n Status: On time");
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoicea2:
					scanf("%c", &menreturn);
					if (menreturn == 'y')
					{
						system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoicea2;
					}
					break;

				case 3:
					system("cls");
					printf("\n Vienna");
					printf("\n Austrian\n");
					printf("\nTERMINAL 2");
					printf("\n OS458");
					printf("\n Departure time 06:00");
					printf("\n Boarding time  03:45");
					printf("\n Friday 23 December 2022");
					printf("\n Expected arrival in Vienna 9:15");
					printf("\n Status: On time");
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoicea3:
					scanf("%c", &menreturn);
					if (menreturn == 'y')
					{
						system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoicea3;
					}
					break;

				case 4:
					system("cls");
					printf("\n Zurich");
					printf("\n Swiss International\n");
					printf("\nTERMINAL 2");
					printf("\n LX345");
					printf("\n Departure time 06:00");
					printf("\n Boarding time  03:45");
					printf("\n Friday 23 December 2022");
					printf("\n Expected arrival in Zurich 8:40");
					printf("\n Status: On time");
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoicea4:
					scanf("%c", &menreturn);
					if (menreturn == 'y')
					{
						system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoicea4;
					}
					break;

				case 5:
					system("cls");
					printf("\n Barcelona");
					printf("\n British Airline\n");
					printf("\nTERMINAL 5");
					printf("\n BA472");
					printf("\n Departure time 06:05");
					printf("\n Boarding time  03:50");
					printf("\n Friday 23 December 2022");
					printf("\n Expected arrival in Barcelona 9:10");
					printf("\n Status: On time");
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoicea5:
					scanf("%c", &menreturn);
					if (menreturn == 'y')
					{
						system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoicea5;
					}
					break;

				case 6:
					system("cls");
					printf("\n Madrid");
					printf("\n British Airline\n");
					printf("\nTERMINAL 3");
					printf("\n BA456");
					printf("\n Departure time 06:05");
					printf("\n Boarding time  03:45");
					printf("\n Friday 23 December 2022");
					printf("\n Expected arrival in Madrid 9:40");
					printf("\n Status: On time");
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoicea69:
					scanf("%c", &menreturn);
					if (menreturn == 'y')
					{
						system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoicea69;
					}
					break;

				case 7:
					system("cls");
					printf("\n Paris");
					printf("\n Air France\n");
					printf("\nTERMINAL 3");
					printf("\n AF1381");
					printf("\n Departure time 06:20");
					printf("\n Boarding time  03:55");
					printf("\n Friday 23 December 2022");
					printf("\n Expected arrival in Paris 8:40");
					printf("\n Status: On time");
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoicea7:
					scanf("%c", &menreturn);
					if (menreturn == 'y')
					{
						system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoicea7;
					}
					break;

				case 8:
					system("cls");
					printf("\n Glasgow");
					printf("\n British Airline\n");
					printf("\n BA1472");
					printf("\n Terminal 5");
					printf("\n Departure time 06:20");
					printf("\n Boarding time  03:45");
					printf("\n Friday 23 December 2022");
					printf("\n Expected arrival in Glasgow 7:50");
					printf("\n Status: On time");
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoicea8:
					scanf("%c", &menreturn);
					if (menreturn == 'y')
					{
						system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoicea8;
					}
					break;

				case 9:
					system("cls");
					printf("\n Amsterdam");
					printf("\n Royal Dutch Airlines\n");
					printf("\nTERMINAL 3");
					printf("\n KL1000");
					printf("\n Departure time 06:30");
					printf("\n Boarding time  03:45");
					printf("\n Friday 23 December 2022");
					printf("\n Expected arrival in Amsterdam 9:00");
					printf("\n Status: On time");
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoicea9:
					scanf("%c", &menreturn);
					if (menreturn == 'y')
					{
						system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoicea9;
					}
					break;

				case 10:
					system("cls");
					printf("\n Warsaw");
					printf("\n LOT Polish Airlines \n");
					printf("\nTERMINAL 2");
					printf("\n LO286");
					printf("\n Departure time 06:30");
					printf("\n Boarding time  04:05");
					printf("\n Friday 23 December 2022");
					printf("\n Expected arrival in Warsaw 10:00");
					printf("\n Status: On time");
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoicea10:
					scanf("%c", &menreturn);
					if (menreturn == 'y')
					{
						system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoicea10;
					}
					break;

				default:
					system("cls");
					printf("Flight not found"); // when invalid choice is entered
					goto menu;
				}
				break;

			case 2: //harcoded data for user panel option2
			areas:
				printf("Select your area: \n");
				printf("\n1.Cape Town ");
				printf("\n2. Lagos ");
				printf("\n3. Johannesburg ");
				printf("\n4. Hong kong ");
				printf("\n5. Perth ");
				printf("\n6. Riyadh ");
				printf("\n7. Singapore ");
				printf("\n8. Accra ");
				printf("\n9. Kuala Lampur");
				printf("\n10. Bridge Town");
				printf("\n11. Toronto ");
				printf("\n12. Dubai ");
				printf("\n13. Dallas ");
				printf("\n14. Kigali ");
				printf("\n15. Bangkok");
				printf("\n16. Tokyo");
				printf("\n17. Doha");
				printf("\n18. Port Louis");
				printf("\n19. Bahrain");
				printf("\n20. Addis Ababa");
				printf("\n21. Dehli");
				printf("\n22. EXIT TO MENU");
				printf("\n    ENTER CORRESPONDING NUMBER    \n");
				scanf("%d", &select);

				switch (select)
				{
				default:
					printf("\nArea unavailable!"); //incase of invalid input
					system("cls");
					goto areas;
					break;

				case 22:
					goto menu; //will take the program execution back to the label menu

				case 1:
					system("cls");
					printf("Cape Town\n");
					printf("\nBritish Airlines");
					printf("\nBA58");
					printf("\nScheduled arrival: 04:45");
					printf("\nTerminal 5");
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoice11:
					scanf("%c", &menreturn);
					if (menreturn == 'y')
					{
						system("cls");
						goto menu;
						break;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						printf("\nInvalid input!\n\n");
						goto retchoice11;
					}
					break;

				case 2:
					system("cls");
					printf("Lagos\n");
					printf("\nBritish Airlines");
					printf("\nExpected Arrival: 04:50");
					printf("\nTerminal 5");
					printf("\nBA74");
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoice219:
					scanf("%c", &menreturn);
					if (menreturn == 'y')
					{
						system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoice219;
					}
				case 3:
					system("cls");
					printf("Hong Kong\n");
					printf("\nCathay Pacific");
					printf("\nCX251");
					printf("\nExpected Arrival:05:00");
					printf("\n4Terminal 3");
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoice31:
					scanf("%c", &menreturn);
					if (menreturn == 'y')
					{
						system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoice31;
					}

				case 4:
					system("cls");
					printf("Perth\n");
					printf("\nQantas");
					printf("\nQFA9 ");
					printf("\nExpected Arrival");
					printf("\nTerminal 3");
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoice41:
					scanf("%c", &menreturn);
					if (menreturn == 'y')
					{
						system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoice41;
					}
				case 5:
					system("cls");
					printf("Riyadh\n");
					printf("\nBritish Airlines");
					printf("\nBA262");
					printf("\nExpected Arrival: 05:20");
					printf("\nTerminal 5");
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoice141:
					scanf("%c", &menreturn);
					if (menreturn == 'y')
					{
						system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoice41;
					}

				case 6:
					system("cls");
					printf("Singapore\n");
					printf("\nBritish Airlines");
					printf("\nBA16");
					printf("\nExpected Arrival: 05:25");
					printf("\nTerminal 5");
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoice61:
					scanf("%c", &menreturn);
					if (menreturn == 'y')
					{
						system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoice61;
					}
					
				case 7:
					system("cls");
					printf("Accra\n");
					printf("\nBritish Airlines");
					printf("\nBA78");
					printf("\nScheduled Arrival: 5:30");
					printf("\nTerminal 5");
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoice71:
					scanf("%c", &menreturn);
					if (menreturn == 'y')
					{
						system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoice71;
					}
					
				case 8:
					system("cls");
					printf("Kuala Lampur\n");
					printf("\nMalayasia Airlines");
					printf("\nMH2 ");
					printf("\nExpected Arrival: 05:35");
					printf("\nTerminal 4");
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoice81:
					scanf("%c", &menreturn);
					if (menreturn == 'y')
					{
						system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoice81;
					}
					
				case 9:
					system("cls");
					printf("Bridge Town\n");
					printf("\nBritish Airlines");
					printf("\nBA254 ");
					printf("\nExpected Arrival: 05:50");
					printf("\nTerminal 5");
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoice91:
					scanf("%c", &menreturn);
					if (menreturn == 'y')
					{
						system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoice91;
					}

				case 10:
					system("cls");
					printf("Toronto\n");
					printf("\nBritish Airlines");
					printf("\nBA92");
					printf("\nExpected Arrival:05:55");
					printf("\nTerminal 5");
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoice101:
					scanf("%c", &menreturn);
					if (menreturn == 'y')
					{
						system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoice101;
					}
					
				case 11:
					system("cls");
					printf("\nDubai");
					printf("\nBritish Airlines");
					printf("\nBA106");
					printf("\nExpected Arrival 06:15");
					printf("\nTerminal 5");
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoice111:
					scanf("%c", &menreturn);
					if (menreturn == 'y')
					{
						system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoice111;
					}
					
				case 12:
					system("cls");
					printf("Dallas");
					printf("\nAmerican Airlines");
					printf("\nAA20");
					printf("\nExpected Arrival: 06:20");
					printf("\nTerminal 4");
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoice121:
					scanf("%c", &menreturn);
					if (menreturn == 'y')
					{
						system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoice121;
					}
					
				case 13:
					system("cls");
					printf("Kigali");
					printf("Rwand Air");
					printf("W8710");
					printf("\nExpected Arrival: 06:20");
					printf("\nTerminal 3");
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoice124:
					scanf("%c", &menreturn);
					if (menreturn == 'y')
					{
						system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoice121;
					}
					
				case 14:
					system("cls");
					printf("Bangkok");
					printf("\nThai Airways");
					printf("\nTG910");
					printf("\nExpected Arrival: 6:20");
					printf("\nTerminal 2");
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoice126:
					scanf("%c", &menreturn);
					if (menreturn == 'y')
					{
						system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoice126;
					}
					
				case 16:
					system("cls");
					printf("Tokyo");
					printf("\nJapan Airlines");
					printf("\nJL410");
					printf("\nExpected Arrival: 6:25");
					printf("\nTerminal 3");
					scanf("%c", &menreturn);
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoice127:
					if (menreturn == 'y')
					{
						system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoice127;
					}

				case 17:
					system("cls");
					printf("Port Louis\n");
					printf("\nAir Mauritius");
					printf("\nMK42");
					printf("\nExpected Arrival: 06:35");
					printf("\nTerminal 4");
					scanf("%c", &menreturn);
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoice128:
					if (menreturn == 'y')
					{
						system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoice128;
					}

				case 18:
					system("cls");
					printf("Bahrain");
					printf("\nGulf Air ");
					printf("\nGF9");
					printf("\nExpected Arrival: 06:35");
					printf("\nTerminal 4");
					scanf("%c", &menreturn);
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoice129:
					if (menreturn == 'y')
					{
						system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoice129;
					}

				case 19:
					system("cls");
					printf("\nAddis Ababa");
					printf("\nEthiopian Airlines ");
					printf("\nET700");
					printf("\nExpected Arrival: 06:35");
					printf("\nTerminal 2");
					scanf("%c", &menreturn);
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoice130:
					if (menreturn == 'y')
					{
						system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoice130;
					}
					
				case 20:
					system("cls");
					printf("\nDelhi");
					printf("\nAir India");
					printf("\nAI161");
					printf("\nExpected Arrival: 06:35");
					printf("\nTerminal 2");

					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoice201:
					scanf("%c", &menreturn);
					if (menreturn == 'y')
					{
						system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoice201;
					}
				}
				break;
			}

		case 4: //connecting flights data

			printf("Enter the corresponding number to the city you are travelling to");
			printf("\n1-Karachi");
			printf("\n2-Qatar");
			printf("\n3-Tokyo");
			printf("\n4-Zurich");
			printf("\n5-Auckland\n");
			scanf("%d", &fln);
			switch (fln)
			{
				
			case 1:
				system("cls");
				printf("Possible connections:\n ");
				printf("\n1.Abu Dhabi-1 hour stay");
				printf("\n2.Dubai-5 hours stay");
				printf("\n3.Doha-35 minutes stay ");
				printf("\n4.Istanbul-12 hours stay");
				printf("\nChoose your connection");
				scanf("%d", &b);
				switch (b)
				{
					
				case 1:
					printf("Successful!\n");
					printf("New flight duration 8hr 09 mins");
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				    retchoicea55:
				    scanf("%c", &menreturn);
					if (menreturn == 'y')
					{
						system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoicea55;
					}

					break;
				case 2:
					printf("Successful!\n");
					printf("New flight duration 1hr 54 mins");
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoicea78:
					scanf("%c", &menreturn);
					if (menreturn == 'y')
					{
						system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoicea78;
					}

					break;
				case 3:
					printf("Successful!\n");
					printf("New flight duration 5hr 11 mins");
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoicea98:
					scanf("%c", &menreturn);
					if (menreturn == 'y')
					{
						system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoicea98;
					}

					break;
				case 4:
					printf("Successful!\n");
					printf("New flight duration 3hr 24 mins");
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoicea87:
					scanf("%c", &menreturn);
					if (menreturn == 'y')
					{
						system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoicea87;
					}

					break;
				}
			case 2:
				system("cls");
				printf("Possible connections:\n ");
				printf("\n1.Abu Dhabi-1 hour stay");
				printf("\n2.Dubai-5 hours stay");
				printf("\n3.Doha-35 minutes stay ");
				printf("\n4.Istanbul-12 hours stay");
				printf("\nChoose your connection\n");
				scanf("%d", &b);
				switch (b)
				{
				case 1:
					printf("Successful!\n");
					printf("New flight duration 13hr 09 mins");
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoicea88:
					scanf("%c", &menreturn);
					if (menreturn == 'y')
					{
						system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoicea88;
					}

					break;
				
				case 2:
					printf("Successful!\n");
					printf("New flight duration 6hr 54 mins");
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoicea778:
					scanf("%c", &menreturn);
					if (menreturn == 'y')
					{
						system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoicea778;
					}

					break;
				
				case 3:
					printf("Successful!\n");
					printf("New flight duration 6hr 17 mins");
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoicea28:
					scanf("%c", &menreturn);
					if (menreturn == 'y')
					{
						+system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoicea28;
					}

					break;
				
				case 4:
					printf("Successful!\n");
					printf("New flight duration 1hr 54 mins");
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoicea718:
					scanf("%c", &menreturn);
					if (menreturn == 'y')
					{
						+system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoicea718;
					}

					break;
				}

			case 3:
				system("cls");
				printf("Possible connections:\n ");
				printf("\n1.Dubai-1 hour stay");
				printf("\n2.Dublin-5 hours stay");
				printf("\n3.Doha-35 minutes stay ");
				printf("\n4.Athens-12 hours stay");
				printf("Choose your connection");
				scanf("%d", &b);
				switch (b)
				{
				
				case 1:
					printf("Successful!\n");
					printf("New flight duration 1hr 044 mins");
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoicea518:
					scanf("%c", &menreturn);
					if (menreturn == 'y')
					{
						+system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoicea518;
					}

					break;
				
				case 2:
					printf("Successful!\n");
					printf("New flight duration 9hr 54 mins");
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoicea878:
					scanf("%c", &menreturn);
					if (menreturn == 'y')
					{
						+system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoicea878;
					}

					break;
				
				case 3:
					printf("Successful!\n");
					printf("New flight duration 5hr 11 mins");
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoicea18:
					scanf("%c", &menreturn);
					if (menreturn == 'y')
					{
						+system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoicea18;
					}

					break;
				
				case 4:
					printf("Successful!\n");
					printf("New flight duration 16hr 01 mins");
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoicea97:
					scanf("%c", &menreturn);
					if (menreturn == 'y')
					{
						+system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoicea97;
					}

					break;
				}
				
			case 5:
				system("cls");
				printf("Possible connections:\n ");
				printf("\n1.Charlotte-1 hour stay");
				printf("\n2.Dallas-5 hours stay");
				printf("\n3.Durham-35 minutes stay ");
				printf("\n4.Newyork-12 hours stay");
				printf("\nChoose your connection");
				scanf("%d", &b);
				switch (b)
				{
				
				case 1:

					printf("Successful!\n");
					printf("New flight duration 8hr 09 mins");
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoicea559:
					scanf("%c", &menreturn);
					if (menreturn == 'y')
					{
						+system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoicea559;
					}

					break;
				
				case 2:
					printf("Successful!\n");
					printf("New flight duration 1hr 54 mins");
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoicea783:
					scanf("%c", &menreturn);
					if (menreturn == 'y')
					{
						+system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoicea783;
					}

					break;
				
				case 3:
					printf("Successful!\n");
					printf("New flight duration 5hr 57 mins");
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoicea981:
					scanf("%c", &menreturn);
					if (menreturn == 'y')
					{
						+system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoicea981;
					}

					break;
				case 4:
					printf("Successful!\n");
					printf("New flight duration 8hr 35 mins");
					printf("\nReturn to menu? (y to return to menu and n to exit)");
				retchoicea827:
					scanf("%c", &menreturn);
					if (menreturn == 'y')
					{
						+system("cls");
						goto menu;
					}
					else if (menreturn == 'n')
					{
						return 0;
					}
					else
					{
						goto retchoicea827;
					}

					break;
				}

				return 0;
			}
		}
	}
}
