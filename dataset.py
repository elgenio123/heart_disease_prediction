

def replace_missing_with_mode(dataframe):
    # Select only numeric columns
    numeric_columns = dataframe.select_dtypes(include=['number']).columns

    # Iterate through each numeric column
    for column in numeric_columns:
        # Calculate the median excluding missing values
        median_value = dataframe[column].mean()

        # Replace missing values in the numeric column with the median
        dataframe[column].fillna(median_value, inplace=True)

    return dataframe



# metadata 
#print(X) 
  
# variable information 
#print(y) 
