User
write a technical design document for this   code                                                                              
  import pandas as pd

file_path = r'Iowa_Liquor_Sales_Sample1.2(cleaned).csv'

# Specify data types for columns 6 (Zip Code) and 14 (Item Number)
column_types = {6: str, 14: str}
df = pd.read_csv(file_path, dtype=column_types)                                                                                                   # Set the display option to show all columns
pd.set_option('display.max_columns', None)
# Display the first few rows of the dataframe
df.head()                                                                                                                                                                                    # Understand the data structure
print("Data Structure:")
print(f"Number of Rows: {df.shape[0]}")
print(f"Number of Columns: {df.shape[1]}")
print("Column Names:", df.columns.tolist())
print("Data Types:")
print(df.dtypes)

# Check for missing values
print("\nMissing Values:")
missing_values = df.isnull().sum()
print(missing_values)

# Basic statistics for numerical features
print("\nBasic Statistical Details:")
basic_stats = df.describe()
print(basic_stats)

# Basic statistics for categorical features
print("\nBasic Statistical Details for Categorical Features:")
categorical_stats = df.describe(include=['O'])  # 'O' indicates 'object' data type
print(categorical_stats)

# Check for duplicate rows
print("\nDuplicate Rows:")
print(f"Number of duplicate rows: {df.duplicated().sum()}")

# Display the first few rows of the dataframe
print("\nFirst Few Rows:")
print(df.head())                                                                                                                                                                    import pandas as pd

# Sample 10,000 random rows from the dataframe
sample_df = df.sample(n=10000, random_state=42)

# Save the sample to a new CSV file 
sample_file_path = 'Iowa_Liquor_Sales_Sample.csv'
sample_df.to_csv(sample_file_path, index=False)

print(sample_df.head())                                                                                                                                                    # Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Downcasting integers and floats
int_columns = ['Store Number', 'Item Number', 'Pack', 'Bottle Volume (ml)', 'Bottles Sold']
float_columns = ['County Number', 'Category', 'Vendor Number', 'State Bottle Cost', 
                 'State Bottle Retail', 'Sale (Dollars)', 'Volume Sold (Liters)', 'Volume Sold (Gallons)']

df[int_columns] = df[int_columns].astype('int32')
df[float_columns] = df[float_columns].astype('float32')                                                                                                                                                        non_int_values = df['Item Number'][~df['Item Number'].apply(lambda x: str(x).isdigit())]
print(non_int_values)                                                                                                                                                      # Remove rows where 'Item Number' is not a digit
df = df[df['Item Number'].apply(lambda x: str(x).isdigit())]

# Attempt conversion again
df[int_columns] = df[int_columns].astype('int32')                                                                                                 # Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Downcasting integers and floats
int_columns = ['Store Number', 'Item Number', 'Pack', 'Bottle Volume (ml)', 'Bottles Sold']
float_columns = ['County Number', 'Category', 'Vendor Number', 'State Bottle Cost', 
                 'State Bottle Retail', 'Sale (Dollars)', 'Volume Sold (Liters)', 'Volume Sold (Gallons)']

df[int_columns] = df[int_columns].astype('int32')
df[float_columns] = df[float_columns].astype('float32')                                                                                  print("Data Types:")
print(df.dtypes)                                                                                                                                                                                           import sys

size_in_bytes = sys.getsizeof(df)
size_in_megabytes = size_in_bytes / (1024**2)  # Convert bytes to megabytes

print(f"Size of DataFrame in memory: {size_in_bytes} bytes")
print(f"Size of DataFrame in memory: {size_in_megabytes} MB")                                                                 # Save the sample to a new CSV file 
file_path = r'./Iowa_Liquor_Sales_Sample1.1(optimizedfiletype).csv'
df.to_csv(file_path, index=False)                                                                                                                                  missing_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
missing_percentage                                                                                                                                                        df.dropna(inplace=True)
# Check for missing values
print("\nMissing Values:")
missing_values = df.isnull().sum()
print(missing_values)                                                                                                                                                        # Save the to a new CSV file 
file_path = r'./Iowa_Liquor_Sales_Sample1.2(cleaned).csv'
df.to_csv(file_path, index=False)                                                                                                                                  df = pd.read_csv(r'./Iowa_Liquor_Sales_Sample1.2(cleaned).csv')                                                               df['Date'] = pd.to_datetime(df['Date'])
df0 = df.copy()                                                                                                                                                                       corr_matrix = df.corr(numeric_only=True)
print(corr_matrix)                                                                                                                                                                  import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


corr_matrix = df.corr(numeric_only=True)

# Set up the matplotlib figure
plt.figure(figsize=(12, 8))

# Draw the heatmap
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm',
            xticklabels=corr_matrix.columns,
            yticklabels=corr_matrix.columns)

# Add title
plt.title('Correlation Matrix Heatmap')

# Show the plot
plt.show()                                                                                                                                                                                   df['Date'].dt                                                                                                                                                                                 # Extract year, month, and day of the week
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.day_name()                                                                                                        df["YM"] = df['Date'].dt.to_period('M').dt.to_timestamp()
df[df["Month"]==7][["Store Number", "YM"]].YM.value_counts().plot()                                                         # Summarize sales by year, month, and day of the week
yearly_sales = df.groupby('Year')['Sale (Dollars)'].sum()
monthly_sales = df.groupby(['Year', 'Month'])['Sale (Dollars)'].sum()
weekday_sales = df.groupby('DayOfWeek')['Sale (Dollars)'].sum()                                                                 # Set display options for pandas
pd.options.display.float_format = '{:,.0f}'.format

print("Yearly Sales:")
print(yearly_sales)

monthly_sales_unstacked = monthly_sales.unstack(level=-1)  # Unstack the month level
print("Monthly Sales:")
print(monthly_sales_unstacked)

print("Sales by Day of the Week:")
print(weekday_sales)                                                                                                                                                        monthly_sales.unstack(level=-1)                                                                                                                               import seaborn as sns
# Plotting July Sales Trend
plt.figure(figsize=(10, 6))
July_sales = monthly_sales.unstack(level=-1)[7]
sns.lineplot(data=July_sales)
plt.title('July Sales Trend(Month-Over-Month)')
plt.xlabel('Year')
plt.ylabel('Total Sales (Dollars)')

# Format the y-axis labels to include commas in numbers
plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

plt.show()                                                                                                                                                                             # Plotting Yearly Sales Trend
plt.figure(figsize=(10, 6))
sns.lineplot(data=yearly_sales)
plt.title('Yearly Sales Trend')
plt.xlabel('Year')
plt.ylabel('Total Sales (Dollars)')

# Format the y-axis labels to include commas in numbers
plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

plt.show()                                                                                                                                                                               # df = df[df['Year'] != 2022]
# Summarize sales by year, month, and day of the week
yearly_sales = df.groupby('Year')['Sale (Dollars)'].sum()
monthly_sales = df.groupby(['Year', 'Month'])['Sale (Dollars)'].sum()
weekday_sales = df.groupby('DayOfWeek')['Sale (Dollars)'].sum()# Set display options for pandas
pd.options.display.float_format = '{:,.0f}'.format                                                                                                        # Plotting Yearly Sales Trend
plt.figure(figsize=(10, 6))
sns.lineplot(data=yearly_sales)
plt.title('Yearly Sales Trend')
plt.xlabel('Year')
plt.ylabel('Total Sales (Dollars)')

# Format the y-axis labels to include commas in numbers
plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

plt.show()                                                                                                                                                                                   monthly_sales_reset = df.groupby(['Year', 'Month'])['Sale (Dollars)'].sum().reset_index()

# Pivot the DataFrame
monthly_sales_pivot = monthly_sales_reset.pivot(index='Month', columns='Year', values='Sale (Dollars)')

# Plotting Stacked Bar Chart
plt.figure(figsize=(12, 8))
monthly_sales_pivot.plot(kind='bar', stacked=True, ax=plt.gca())

plt.title('Monthly Sales Trend by Year (Stacked)')
plt.xlabel('Month')
plt.ylabel('Total Sales (Dollars)')
plt.xticks(range(0, 12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=0)

# Format the y-axis labels to include commas in numbers
plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

# Place the legend outside the graph
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()                                                                                                                                                                                 import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

product_sales = df.groupby('Item Description')['Sale (Dollars)'].sum().sort_values(ascending=False)
top_products = product_sales.head(40)

# Plotting the top products
plt.figure(figsize=(12, 8))
sns.barplot(x=top_products.values, y=top_products.index)

# Set the x-axis formatter to include commas
plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

plt.title('Top 40 Most Popular Products by Sales')
plt.xlabel('Total Sales (Dollars)')
plt.ylabel('Product')
plt.show()                                                                                                                                                                                # Aggregate sales by category
category_sales = df.groupby('Category Name')['Sale (Dollars)'].sum().sort_values(ascending=False)

# Select top categories for visualization 
top_categories = category_sales.head(40)

# Plotting sales by category
plt.figure(figsize=(12, 8))
sns.barplot(x=top_categories.values, y=top_categories.index)

# Set the x-axis formatter to include commas
plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
plt.title('Top Categories by Sales')
plt.xlabel('Total Sales (Dollars)')
plt.ylabel('Category')
plt.show()                                                                                                                                                                                # Aggregate sales by store
store_sales = df.groupby('Store Name')['Sale (Dollars)'].sum().sort_values(ascending=False)

# Select top 40 stores
top_stores = store_sales.head(40)

# Plotting the top stores
plt.figure(figsize=(15, 10))
sns.barplot(x=top_stores.values, y=top_stores.index)

# Format the x-axis labels to include commas in numbers
plt.gca().get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

plt.title('Top 40 Stores by Sales')
plt.xlabel('Total Sales (Dollars)')
plt.ylabel('Store Name')
plt.show()                                                                                                                                                                                # Aggregate sales by city
city_sales = df.groupby('City')['Sale (Dollars)'].sum().sort_values(ascending=False)

# Select top 40 cities
top_cities = city_sales.head(40)

# Plotting the top cities
plt.figure(figsize=(15, 10))
sns.barplot(x=top_cities.values, y=top_cities.index)

# Format the x-axis labels to include commas in numbers
plt.gca().get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

plt.title('Top 40 Cities by Sales')
plt.xlabel('Total Sales (Dollars)')
plt.ylabel('City')
plt.show()                                                                                                                                                                                     # Count the occurrences of each bottle volume
bottle_volume_counts = df['Bottle Volume (ml)'].value_counts().head(15)

# Plotting the most popular bottle volumes
plt.figure(figsize=(12, 8))
sns.barplot(x=bottle_volume_counts.values, y=bottle_volume_counts.index.astype(str))

# Set the x-axis formatter to include commas
plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

plt.title('Top 15 Most Popular Bottle Volumes')
plt.xlabel('Number of Occurrences')
plt.ylabel('Bottle Volume (ml)')
plt.show()                                                                                                                                                                                    import pandas as pd

file_path = r'./Iowa_Liquor_Sales_Sample1.1(optimizedfiletype).csv'
df = pd.read_csv(file_path)                                                                                                                                              # Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Downcasting integers and floats
int_columns = ['Store Number', 'Item Number', 'Pack', 'Bottle Volume (ml)', 'Bottles Sold']
float_columns = ['County Number', 'Category', 'Vendor Number', 'State Bottle Cost', 
                 'State Bottle Retail', 'Sale (Dollars)', 'Volume Sold (Liters)', 'Volume Sold (Gallons)']

df[int_columns] = df[int_columns].astype('int32')
df[float_columns] = df[float_columns].astype('float32')                                                                                      print("Data Types:")
print(df.dtypes)                                                                                                                                                                        # Set the display option to show all columns
pd.set_option('display.max_columns', None)
# Display the first few rows of the dataframe
df.head()                                                                                                                                                                                         # Calculate the profit for each product
df['Profit'] = (df['State Bottle Retail'] - df['State Bottle Cost']) * df['Bottles Sold']

# Group by Item Number or Item Description and sum the profit
grouped_data = df.groupby(['Item Description'])['Profit'].sum().reset_index()

# Sort the products by profit in descending order and get the top 100
top_100_products = grouped_data.sort_values(by='Profit', ascending=False).head(100)

top_100_products.head(100)                                                                                                                                                  print("Data Structure:")
print(f"Number of Rows: {top_100_products.shape[0]}")
print(f"Number of Columns: {top_100_products.shape[1]}")
unique_values = top_100_products['Item Description'].nunique()
print(f"Number of unique values in 'Item Description': {unique_values}")                                                  # Step 1: Create a list of Item Description from top_100_products
top_100_items = top_100_products['Item Description'].tolist()

# Step 2: Filter the original dataframe
filtered_df = df[df['Item Description'].isin(top_100_items)]

# Display the first few rows of the filtered dataframe
filtered_df.head()                                                                                                                                                                    print("Data Structure:")
print(f"Number of Rows: {filtered_df.shape[0]}")
print(f"Number of Columns: {filtered_df.shape[1]}")
unique_values = filtered_df['Item Description'].nunique()
print(f"Number of unique values in 'Item Description': {unique_values}")                                                    filtered_df['Year'] = filtered_df['Date'].dt.year
filtered_df['Month'] = filtered_df['Date'].dt.month
filtered_df['DayOfWeek'] = filtered_df['Date'].dt.day_name()                                                                                file_path = r'./top100.csv'
filtered_df.to_csv(file_path, index=False)                                                                                                                      import pandas as pd
import matplotlib.pyplot as plt

file_path = r'./top100.csv'

column_types = {6: str, 14: str}
df = pd.read_csv(file_path, dtype=column_types)

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Downcasting integers and floats
int_columns = ['Store Number', 'Item Number', 'Pack', 'Bottle Volume (ml)', 'Bottles Sold']
float_columns = ['County Number', 'Category', 'Vendor Number', 'State Bottle Cost', 
                 'State Bottle Retail', 'Sale (Dollars)', 'Volume Sold (Liters)', 'Volume Sold (Gallons)']

df[int_columns] = df[int_columns].astype('int32')
df[float_columns] = df[float_columns].astype('float32')                                                                                     df0 = df.copy()                                                                                                                                                                           df                                                                                                                                                                                                        df["YM"] = df['Date'].dt.to_period('M').dt.to_timestamp()
df[["Store Number", "YM"]].YM.value_counts().plot()                                                                                                       top10_store = df.groupby(['Store Number'])['Sale (Dollars)'].sum().sort_values(ascending=False).head(10).reset_index()["Store Number"]
top10_store                                                                                                                                                                                   df[df["Store Number"].isin(top10_store)].groupby(['YM', 'Store Number'])['Sale (Dollars)'].sum().reset_index()
import seaborn as sns
sns.lineplot(data=flights, x="year", y="passengers", hue="month")                                                                       # Understand the data structure
print("Data Structure:")
print(f"Number of Rows: {df.shape[0]}")
print(f"Number of Columns: {df.shape[1]}")
print("Column Names:", df.columns.tolist())
print("Data Types:")
print(df.dtypes)

df.dropna(inplace=True)
# Check for missing values
print("\nMissing Values:")
missing_values = df.isnull().sum()
print(missing_values)

# Basic statistics for numerical features
print("\nBasic Statistical Details:")
basic_stats = df.describe()
print(basic_stats)

# Basic statistics for categorical features
print("\nBasic Statistical Details for Categorical Features:")
categorical_stats = df.describe(include=['O'])  # 'O' indicates 'object' data type
print(categorical_stats)

# Check for duplicate rows
print("\nDuplicate Rows:")
print(f"Number of duplicate rows: {df.duplicated().sum()}")

# Display the first few rows of the dataframe
print("\nFirst Few Rows:")
print(df.head())                                                                                                                                                                           # Extract the year from 'Date' column
df = df0.copy()  # df = df[df['Year'] != 2022] 
df['Year'] = df['Date'].dt.year

# Sum of sales for each item description and year
sales_sum = df.groupby(['Item Description', 'Year'])['Sale (Dollars)'].sum().reset_index()

# Identify top 10 items overall based on total sales
top_items_overall = sales_sum.groupby('Item Description')['Sale (Dollars)'].sum().nlargest(10).index

# Filter the data to keep only the top 10 items
filtered_data = sales_sum[sales_sum['Item Description'].isin(top_items_overall)]

# Pivot the data for plotting
pivot_data = filtered_data.pivot(index='Year', columns='Item Description', values='Sale (Dollars)')

# Plotting the trend lines
plt.figure(figsize=(15, 8))
markers = ['o', 's', 'v', '^', 'D', 'p', '*', 'h', 'x', '+']  # List of markers
for column, marker in zip(pivot_data.columns, markers):
    plt.plot(pivot_data.index, pivot_data[column], marker=marker, linewidth=2, label=column) 

plt.axhline(0, color='black', linestyle='--')  # Add a zero line to the y-axis

plt.title('Annual Sales Trend of Top 10 Items')
plt.xlabel('Year')
plt.ylabel('Total Sales ($)')
plt.xticks(pivot_data.index)  # Set the x-axis ticks to be the years
plt.legend(title='Item Description', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
plt.grid(True)
plt.show()                                                                                                                                                                                         # df = df[df['Year'] != 2022]
# Calculate the sum of sales for each item and year
sales_sum = df.groupby(['Item Description', 'Year'])['Sale (Dollars)'].sum().reset_index()

# Identify items ranked 11th to 21st based on total sales
items_11_to_21 = sales_sum.groupby('Item Description')['Sale (Dollars)'].sum().nlargest(21).index[10:]

# Filter the data to keep only items ranked 11th to 21st
filtered_data = sales_sum[sales_sum['Item Description'].isin(items_11_to_21)]

# Pivot the data for plotting
pivot_data = filtered_data.pivot(index='Year', columns='Item Description', values='Sale (Dollars)')

# Plotting the trend lines
plt.figure(figsize=(15, 8))
markers = ['o', 's', 'v', '^', 'D', 'p', '*', 'h', 'x', '+']  
for column, marker in zip(pivot_data.columns, markers):
    plt.plot(pivot_data.index, pivot_data[column], marker=marker, linewidth=2, label=column) 

plt.axhline(0, color='black', linestyle='--')  # Add a zero line to the y-axis

plt.title('Annual Sales Trend of Items Ranked 11th to 21st')
plt.xlabel('Year')
plt.ylabel('Total Sales ($)')
plt.xticks(pivot_data.index)  # Set the x-axis ticks to be the years
plt.legend(title='Item Description', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
plt.grid(True)
plt.show()                                                                                                                                                                                       # Identify items ranked 22nd to 31st based on total sales
items_22_to_31 = sales_sum.groupby('Item Description')['Sale (Dollars)'].sum().nlargest(31).index[21:]

# Filter the data to keep only items ranked 22nd to 31st
filtered_data = sales_sum[sales_sum['Item Description'].isin(items_22_to_31)]

# Pivot the data for plotting
pivot_data = filtered_data.pivot(index='Year', columns='Item Description', values='Sale (Dollars)')

# Plotting the trend lines
plt.figure(figsize=(15, 8))
markers = ['o', 's', 'v', '^', 'D', 'p', '*', 'h', 'x', '+']  # You might need different markers for these items
for column, marker in zip(pivot_data.columns, markers):
    plt.plot(pivot_data.index, pivot_data[column], marker=marker, linewidth=2, label=column) 

plt.axhline(0, color='black', linestyle='--')  # Add a zero line to the y-axis

plt.title('Annual Sales Trend of Items Ranked 22nd to 31st')
plt.xlabel('Year')
plt.ylabel('Total Sales ($)')
plt.xticks(pivot_data.index)  # Set the x-axis ticks to be the years
plt.legend(title='Item Description', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
plt.grid(True)
plt.show()                                                                                                                                                                                          import matplotlib.pyplot as plt


# Identify items ranked 32nd to 41st based on total sales
items_32_to_41 = sales_sum.groupby('Item Description')['Sale (Dollars)'].sum().nlargest(41).index[31:]

# Filter the data to keep only items ranked 32nd to 41st
filtered_data = sales_sum[sales_sum['Item Description'].isin(items_32_to_41)]

# Pivot the data for plotting
pivot_data = filtered_data.pivot(index='Year', columns='Item Description', values='Sale (Dollars)')

# Plotting the trend lines
plt.figure(figsize=(15, 8))
markers = ['o', 's', 'v', '^', 'D', 'p', '*', 'h', 'x', '+']  # Adjust markers as needed
for column, marker in zip(pivot_data.columns, markers):
    plt.plot(pivot_data.index, pivot_data[column], marker=marker, linewidth=2, label=column) 

plt.axhline(0, color='black', linestyle='--')  # Add a zero line to the y-axis

plt.title('Annual Sales Trend of Items Ranked 32nd to 41st')
plt.xlabel('Year')
plt.ylabel('Total Sales ($)')
plt.xticks(pivot_data.index)  # Set the x-axis ticks to be the years
plt.legend(title='Item Description', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
plt.grid(True)
plt.show()                                                                                                                                                                                           import matplotlib.pyplot as plt

# Identify items ranked 42nd to 51st based on total sales
items_42_to_51 = sales_sum.groupby('Item Description')['Sale (Dollars)'].sum().nlargest(51).index[41:]

# Filter the data to keep only items ranked 42nd to 51st
filtered_data = sales_sum[sales_sum['Item Description'].isin(items_42_to_51)]

# Pivot the data for plotting
pivot_data = filtered_data.pivot(index='Year', columns='Item Description', values='Sale (Dollars)')

# Plotting the trend lines
plt.figure(figsize=(15, 8))
markers = ['o', 's', 'v', '^', 'D', 'p', '*', 'h', 'x', '+']  # Adjust markers as needed
for column, marker in zip(pivot_data.columns, markers):
    plt.plot(pivot_data.index, pivot_data[column], marker=marker, linewidth=2, label=column) 

plt.axhline(0, color='black', linestyle='--')  # Add a zero line to the y-axis

plt.title('Annual Sales Trend of Items Ranked 42nd to 51st')
plt.xlabel('Year')
plt.ylabel('Total Sales ($)')
plt.xticks(pivot_data.index)  # Set the x-axis ticks to be the years
plt.legend(title='Item Description', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
plt.grid(True)
plt.show()                                                                                                                                                                                        import matplotlib.pyplot as plt

# Identify items ranked 52nd to 61st based on total sales
items_52_to_61 = sales_sum.groupby('Item Description')['Sale (Dollars)'].sum().nlargest(61).index[51:]

# Filter the data to keep only items ranked 52nd to 61st
filtered_data = sales_sum[sales_sum['Item Description'].isin(items_52_to_61)]

# Pivot the data for plotting
pivot_data = filtered_data.pivot(index='Year', columns='Item Description', values='Sale (Dollars)')

# Plotting the trend lines
plt.figure(figsize=(15, 8))
markers = ['o', 's', 'v', '^', 'D', 'p', '*', 'h', 'x', '+'] 
for column, marker in zip(pivot_data.columns, markers):
    plt.plot(pivot_data.index, pivot_data[column], marker=marker, linewidth=2, label=column) 

plt.axhline(0, color='black', linestyle='--')  # Add a zero line to the y-axis

plt.title('Annual Sales Trend of Items Ranked 52nd to 61st')
plt.xlabel('Year')
plt.ylabel('Total Sales ($)')
plt.xticks(pivot_data.index)  # Set the x-axis ticks to be the years
plt.legend(title='Item Description', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
plt.grid(True)
plt.show()                                                                                                                                                                                            import matplotlib.pyplot as plt

# Identify items ranked 62nd to 71st based on total sales
items_62_to_71 = sales_sum.groupby('Item Description')['Sale (Dollars)'].sum().nlargest(71).index[61:]

# Filter the data to keep only items ranked 62nd to 71st
filtered_data = sales_sum[sales_sum['Item Description'].isin(items_62_to_71)]

# Pivot the data for plotting
pivot_data = filtered_data.pivot(index='Year', columns='Item Description', values='Sale (Dollars)')

# Plotting the trend lines
plt.figure(figsize=(15, 8))
markers = ['o', 's', 'v', '^', 'D', 'p', '*', 'h', 'x', '+'] 
for column, marker in zip(pivot_data.columns, markers):
    plt.plot(pivot_data.index, pivot_data[column], marker=marker, linewidth=2, label=column) 

plt.axhline(0, color='black', linestyle='--')  # Add a zero line to the y-axis

plt.title('Annual Sales Trend of Items Ranked 62nd to 71st')
plt.xlabel('Year')
plt.ylabel('Total Sales ($)')
plt.xticks(pivot_data.index)  # Set the x-axis ticks to be the years
plt.legend(title='Item Description', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
plt.grid(True)
plt.show()                                                                                                                                                                                            import matplotlib.pyplot as plt

# Identify items ranked 72nd to 81st based on total sales
items_72_to_81 = sales_sum.groupby('Item Description')['Sale (Dollars)'].sum().nlargest(81).index[71:]

# Filter the data to keep only items ranked 72nd to 81st
filtered_data = sales_sum[sales_sum['Item Description'].isin(items_72_to_81)]

# Pivot the data for plotting
pivot_data = filtered_data.pivot(index='Year', columns='Item Description', values='Sale (Dollars)')

# Plotting the trend lines
plt.figure(figsize=(15, 8))
markers = ['o', 's', 'v', '^', 'D', 'p', '*', 'h', 'x', '+']  
for column, marker in zip(pivot_data.columns, markers):
    plt.plot(pivot_data.index, pivot_data[column], marker=marker, linewidth=2, label=column) 

plt.axhline(0, color='black', linestyle='--')  # Add a zero line to the y-axis

plt.title('Annual Sales Trend of Items Ranked 72nd to 81st')
plt.xlabel('Year')
plt.ylabel('Total Sales ($)')
plt.xticks(pivot_data.index)  # Set the x-axis ticks to be the years
plt.legend(title='Item Description', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
plt.grid(True)
plt.show()                                                                                                                                                                                             import matplotlib.pyplot as plt

# Identify items ranked 82nd to 91st based on total sales
items_82_to_91 = sales_sum.groupby('Item Description')['Sale (Dollars)'].sum().nlargest(91).index[81:]

# Filter the data to keep only items ranked 82nd to 91st
filtered_data = sales_sum[sales_sum['Item Description'].isin(items_82_to_91)]

# Pivot the data for plotting
pivot_data = filtered_data.pivot(index='Year', columns='Item Description', values='Sale (Dollars)')

# Plotting the trend lines
plt.figure(figsize=(15, 8))
markers = ['o', 's', 'v', '^', 'D', 'p', '*', 'h', 'x', '+']  # Adjust markers as needed
for column, marker in zip(pivot_data.columns, markers):
    plt.plot(pivot_data.index, pivot_data[column], marker=marker, linewidth=2, label=column) 

plt.axhline(0, color='black', linestyle='--')  # Add a zero line to the y-axis

plt.title('Annual Sales Trend of Items Ranked 82nd to 91st')
plt.xlabel('Year')
plt.ylabel('Total Sales ($)')
plt.xticks(pivot_data.index)  # Set the x-axis ticks to be the years
plt.legend(title='Item Description', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
plt.grid(True)
plt.show()                                                                                                                                                                                                 import matplotlib.pyplot as plt

# Identify items ranked 92nd to 100th based on total sales
items_92_to_100 = sales_sum.groupby('Item Description')['Sale (Dollars)'].sum().nlargest(100).index[91:]

# Filter the data to keep only items ranked 92nd to 100th
filtered_data = sales_sum[sales_sum['Item Description'].isin(items_92_to_100)]

# Pivot the data for plotting
pivot_data = filtered_data.pivot(index='Year', columns='Item Description', values='Sale (Dollars)')

# Plotting the trend lines
plt.figure(figsize=(15, 8))
markers = ['o', 's', 'v', '^', 'D', 'p', '*', 'h', 'x', '+']  # Adjust markers as needed, add more if necessary
for column, marker in zip(pivot_data.columns, markers):
    plt.plot(pivot_data.index, pivot_data[column], marker=marker, linewidth=2, label=column) 

plt.axhline(0, color='black', linestyle='--')  # Add a zero line to the y-axis

plt.title('Annual Sales Trend of Items Ranked 92nd to 100th')
plt.xlabel('Year')
plt.ylabel('Total Sales ($)')
plt.xticks(pivot_data.index)  # Set the x-axis ticks to be the years
plt.legend(title='Item Description', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
plt.grid(True)
plt.show()                                                                                                                                                                                      # First, find the 'Item Description' values that have no data for 2021
items_with_no_2021_data = df[df['Year'] == 2021]['Item Description'].unique()

# Then, filter the DataFrame to exclude these items
df = df[~df['Item Description'].isin(items_with_no_2021_data)]                                                                          print("Data Structure:")
print(f"Number of Rows: {df.shape[0]}")
print(f"Number of Columns: {df.shape[1]}")
unique_values = df['Item Description'].nunique()
print(f"Number of unique values in 'Item Description': {unique_values}")                                                                import pandas as pd
import matplotlib.pyplot as plt

file_path = r'G:\My Drive\Georgian assignments\2 semester\Social Data Mining Techniques\Final project IOWA\Iowa_Liquor_Sales_Sample1.1(optimizedfiletype).csv'

column_types = {6: str, 14: str}
df = pd.read_csv(file_path, dtype=column_types)

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Downcasting integers and floats
int_columns = ['Store Number', 'Item Number', 'Pack', 'Bottle Volume (ml)', 'Bottles Sold']
float_columns = ['County Number', 'Category', 'Vendor Number', 'State Bottle Cost', 
                 'State Bottle Retail', 'Sale (Dollars)', 'Volume Sold (Liters)', 'Volume Sold (Gallons)']

df[int_columns] = df[int_columns].astype('int32')
df[float_columns] = df[float_columns].astype('float32')                                                                                    # Understand the data structure
print("Data Structure:")
print(f"Number of Rows: {df.shape[0]}")
print(f"Number of Columns: {df.shape[1]}")                                                                                                                      df['Year'] = pd.to_datetime(df['Date']).dt.year

# identify products with sales in 2021
products_with_2021_sales = df[df['Year'] == 2021]['Item Description'].unique()

# Filter the DataFrame to include only these products and create a copy
df_2021 = df[df['Item Description'].isin(products_with_2021_sales)].copy()

# Calculate the profit for each product
df_2021['Profit'] = (df_2021['State Bottle Retail'] - df_2021['State Bottle Cost']) * df_2021['Bottles Sold']

# Group by Item Description and sum the profit
grouped_data = df_2021.groupby('Item Description')['Profit'].sum().reset_index()

# Sort the products by profit in descending order and get the top 100
top_100_products = grouped_data.sort_values(by='Profit', ascending=False).head(100)

pd.set_option('display.float_format', '{:,.2f}'.format)
top_100_products.head(100)                                                                                                                                                        print("Data Structure:")
print(f"Number of Rows: {top_100_products.shape[0]}")
print(f"Number of Columns: {top_100_products.shape[1]}")
unique_values = df['Item Description'].nunique()
print(f"Number of unique values in 'Item Description': {unique_values}")                                                          # Step 1: Create a list of Item Description from top_100_products
top_100_items = top_100_products['Item Description'].tolist()

# Step 2: Filter the original dataframe
filtered_df = df[df['Item Description'].isin(top_100_items)]

# Display the first few rows of the filtered dataframe
filtered_df.head()                                                                                                                                                                            # Understand the data structure
print("Data Structure:")
print(f"Number of Rows: {filtered_df.shape[0]}")
print(f"Number of Columns: {filtered_df.shape[1]}")
print("Column Names:", filtered_df.columns.tolist())
print("Data Types:")
print(filtered_df.dtypes)

filtered_df.dropna(inplace=True)
# Check for missing values
print("\nMissing Values:")
missing_values = filtered_df.isnull().sum()
print(missing_values)

# Basic statistics for numerical features
print("\nBasic Statistical Details:")
basic_stats = filtered_df.describe()
print(basic_stats)

# Basic statistics for categorical features
print("\nBasic Statistical Details for Categorical Features:")
categorical_stats = filtered_df.describe(include=['O'])  # 'O' indicates 'object' data type
print(categorical_stats)

# Check for duplicate rows
print("\nDuplicate Rows:")
print(f"Number of duplicate rows: {filtered_df.duplicated().sum()}")

# Display the first few rows of the dataframe
print("\nFirst Few Rows:")
print(filtered_df.head())                                                                                                                                                          import pandas as pd
import matplotlib.pyplot as plt

file_path = r'G:\My Drive\Georgian assignments\2 semester\Social Data Mining Techniques\Final project IOWA\top100(2021filtered).csv'

column_types = {6: str, 14: str}
df = pd.read_csv(file_path, dtype=column_types)

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Downcasting integers and floats
int_columns = ['Store Number', 'Item Number', 'Pack', 'Bottle Volume (ml)', 'Bottles Sold']
float_columns = ['County Number', 'Category', 'Vendor Number', 'State Bottle Cost', 
                 'State Bottle Retail', 'Sale (Dollars)', 'Volume Sold (Liters)', 'Volume Sold (Gallons)']

df[int_columns] = df[int_columns].astype('int32')
df[float_columns] = df[float_columns].astype('float32')

# Extract the year from 'Date' column
df = df[df['Year'] != 2022]
df['Year'] = df['Date'].dt.year

# Understand the data structure
print("Data Structure:")
print(f"Number of Rows: {df.shape[0]}")
print(f"Number of Columns: {df.shape[1]}")
print("Column Names:", df.columns.tolist())
print("Data Types:")
print(df.dtypes)

df.dropna(inplace=True)
# Check for missing values
print("\nMissing Values:")
missing_values = df.isnull().sum()
print(missing_values)

# Basic statistics for numerical features
print("\nBasic Statistical Details:")
basic_stats = df.describe()
print(basic_stats)

# Basic statistics for categorical features
print("\nBasic Statistical Details for Categorical Features:")
categorical_stats = df.describe(include=['O'])  # 'O' indicates 'object' data type
print(categorical_stats)

# Check for duplicate rows
print("\nDuplicate Rows:")
print(f"Number of duplicate rows: {df.duplicated().sum()}")

# Display the first few rows of the dataframe
print("\nFirst Few Rows:")
print(filtered_df.head())                                                                                                                                                      # Sum of sales for each item description and year
sales_sum = df.groupby(['Item Description', 'Year'])['Sale (Dollars)'].sum().reset_index()

# Identify top 10 items overall based on total sales
top_items_overall = sales_sum.groupby('Item Description')['Sale (Dollars)'].sum().nlargest(10).index

# Filter the data to keep only the top 10 items
filtered_data = sales_sum[sales_sum['Item Description'].isin(top_items_overall)]

# Pivot the data for plotting
pivot_data = filtered_data.pivot(index='Year', columns='Item Description', values='Sale (Dollars)')

# Plotting the trend lines
plt.figure(figsize=(15, 8))
markers = ['o', 's', 'v', '^', 'D', 'p', '*', 'h', 'x', '+']  # List of markers
for column, marker in zip(pivot_data.columns, markers):
    plt.plot(pivot_data.index, pivot_data[column], marker=marker, linewidth=2, label=column) 

plt.axhline(0, color='black', linestyle='--')  # Add a zero line to the y-axis

plt.title('Annual Sales Trend of Top 10 Items')
plt.xlabel('Year')
plt.ylabel('Total Sales ($)')
plt.xticks(pivot_data.index)  # Set the x-axis ticks to be the years
plt.legend(title='Item Description', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
plt.grid(True)
plt.show()                                                                                                                                                                                         df = df[df['Year'] != 2022]
# Calculate the sum of sales for each item and year
sales_sum = df.groupby(['Item Description', 'Year'])['Sale (Dollars)'].sum().reset_index()

# Identify items ranked 11th to 21st based on total sales
items_11_to_21 = sales_sum.groupby('Item Description')['Sale (Dollars)'].sum().nlargest(21).index[10:]

# Filter the data to keep only items ranked 11th to 21st
filtered_data = sales_sum[sales_sum['Item Description'].isin(items_11_to_21)]

# Pivot the data for plotting
pivot_data = filtered_data.pivot(index='Year', columns='Item Description', values='Sale (Dollars)')

# Plotting the trend lines
plt.figure(figsize=(15, 8))
markers = ['o', 's', 'v', '^', 'D', 'p', '*', 'h', 'x', '+']  
for column, marker in zip(pivot_data.columns, markers):
    plt.plot(pivot_data.index, pivot_data[column], marker=marker, linewidth=2, label=column) 

plt.axhline(0, color='black', linestyle='--')  # Add a zero line to the y-axis

plt.title('Annual Sales Trend of Items Ranked 11th to 21st')
plt.xlabel('Year')
plt.ylabel('Total Sales ($)')
plt.xticks(pivot_data.index)  # Set the x-axis ticks to be the years
plt.legend(title='Item Description', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
plt.grid(True)
plt.show()                                                                                                                                                                                     # Identify items ranked 22nd to 31st based on total sales
items_22_to_31 = sales_sum.groupby('Item Description')['Sale (Dollars)'].sum().nlargest(31).index[21:]

# Filter the data to keep only items ranked 22nd to 31st
filtered_data = sales_sum[sales_sum['Item Description'].isin(items_22_to_31)]

# Pivot the data for plotting
pivot_data = filtered_data.pivot(index='Year', columns='Item Description', values='Sale (Dollars)')

# Plotting the trend lines
plt.figure(figsize=(15, 8))
markers = ['o', 's', 'v', '^', 'D', 'p', '*', 'h', 'x', '+']  # You might need different markers for these items
for column, marker in zip(pivot_data.columns, markers):
    plt.plot(pivot_data.index, pivot_data[column], marker=marker, linewidth=2, label=column) 

plt.axhline(0, color='black', linestyle='--')  # Add a zero line to the y-axis

plt.title('Annual Sales Trend of Items Ranked 22nd to 31st')
plt.xlabel('Year')
plt.ylabel('Total Sales ($)')
plt.xticks(pivot_data.index)  # Set the x-axis ticks to be the years
plt.legend(title='Item Description', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
plt.grid(True)
plt.show()                                                                                                                                                                                           import matplotlib.pyplot as plt


# Identify items ranked 32nd to 41st based on total sales
items_32_to_41 = sales_sum.groupby('Item Description')['Sale (Dollars)'].sum().nlargest(41).index[31:]

# Filter the data to keep only items ranked 32nd to 41st
filtered_data = sales_sum[sales_sum['Item Description'].isin(items_32_to_41)]

# Pivot the data for plotting
pivot_data = filtered_data.pivot(index='Year', columns='Item Description', values='Sale (Dollars)')

# Plotting the trend lines
plt.figure(figsize=(15, 8))
markers = ['o', 's', 'v', '^', 'D', 'p', '*', 'h', 'x', '+']  # Adjust markers as needed
for column, marker in zip(pivot_data.columns, markers):
    plt.plot(pivot_data.index, pivot_data[column], marker=marker, linewidth=2, label=column) 

plt.axhline(0, color='black', linestyle='--')  # Add a zero line to the y-axis

plt.title('Annual Sales Trend of Items Ranked 32nd to 41st')
plt.xlabel('Year')
plt.ylabel('Total Sales ($)')
plt.xticks(pivot_data.index)  # Set the x-axis ticks to be the years
plt.legend(title='Item Description', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
plt.grid(True)
plt.show()                                                                                                                                                                                    import matplotlib.pyplot as plt

# Identify items ranked 42nd to 51st based on total sales
items_42_to_51 = sales_sum.groupby('Item Description')['Sale (Dollars)'].sum().nlargest(51).index[41:]

# Filter the data to keep only items ranked 42nd to 51st
filtered_data = sales_sum[sales_sum['Item Description'].isin(items_42_to_51)]

# Pivot the data for plotting
pivot_data = filtered_data.pivot(index='Year', columns='Item Description', values='Sale (Dollars)')

# Plotting the trend lines
plt.figure(figsize=(15, 8))
markers = ['o', 's', 'v', '^', 'D', 'p', '*', 'h', 'x', '+']  # Adjust markers as needed
for column, marker in zip(pivot_data.columns, markers):
    plt.plot(pivot_data.index, pivot_data[column], marker=marker, linewidth=2, label=column) 

plt.axhline(0, color='black', linestyle='--')  # Add a zero line to the y-axis

plt.title('Annual Sales Trend of Items Ranked 42nd to 51st')
plt.xlabel('Year')
plt.ylabel('Total Sales ($)')
plt.xticks(pivot_data.index)  # Set the x-axis ticks to be the years
plt.legend(title='Item Description', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
plt.grid(True)
plt.show()                                                                                                                                                                                             import matplotlib.pyplot as plt

# Identify items ranked 52nd to 61st based on total sales
items_52_to_61 = sales_sum.groupby('Item Description')['Sale (Dollars)'].sum().nlargest(61).index[51:]

# Filter the data to keep only items ranked 52nd to 61st
filtered_data = sales_sum[sales_sum['Item Description'].isin(items_52_to_61)]

# Pivot the data for plotting
pivot_data = filtered_data.pivot(index='Year', columns='Item Description', values='Sale (Dollars)')

# Plotting the trend lines
plt.figure(figsize=(15, 8))
markers = ['o', 's', 'v', '^', 'D', 'p', '*', 'h', 'x', '+'] 
for column, marker in zip(pivot_data.columns, markers):
    plt.plot(pivot_data.index, pivot_data[column], marker=marker, linewidth=2, label=column) 

plt.axhline(0, color='black', linestyle='--')  # Add a zero line to the y-axis

plt.title('Annual Sales Trend of Items Ranked 52nd to 61st')
plt.xlabel('Year')
plt.ylabel('Total Sales ($)')
plt.xticks(pivot_data.index)  # Set the x-axis ticks to be the years
plt.legend(title='Item Description', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
plt.grid(True)
plt.show()                                                                                                                                                                                      import matplotlib.pyplot as plt

# Identify items ranked 62nd to 71st based on total sales
items_62_to_71 = sales_sum.groupby('Item Description')['Sale (Dollars)'].sum().nlargest(71).index[61:]

# Filter the data to keep only items ranked 62nd to 71st
filtered_data = sales_sum[sales_sum['Item Description'].isin(items_62_to_71)]

# Pivot the data for plotting
pivot_data = filtered_data.pivot(index='Year', columns='Item Description', values='Sale (Dollars)')

# Plotting the trend lines
plt.figure(figsize=(15, 8))
markers = ['o', 's', 'v', '^', 'D', 'p', '*', 'h', 'x', '+'] 
for column, marker in zip(pivot_data.columns, markers):
    plt.plot(pivot_data.index, pivot_data[column], marker=marker, linewidth=2, label=column) 

plt.axhline(0, color='black', linestyle='--')  # Add a zero line to the y-axis

plt.title('Annual Sales Trend of Items Ranked 62nd to 71st')
plt.xlabel('Year')
plt.ylabel('Total Sales ($)')
plt.xticks(pivot_data.index)  # Set the x-axis ticks to be the years
plt.legend(title='Item Description', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
plt.grid(True)
plt.show()                                                                                                                                                                                         import matplotlib.pyplot as plt

# Identify items ranked 72nd to 81st based on total sales
items_72_to_81 = sales_sum.groupby('Item Description')['Sale (Dollars)'].sum().nlargest(81).index[71:]

# Filter the data to keep only items ranked 72nd to 81st
filtered_data = sales_sum[sales_sum['Item Description'].isin(items_72_to_81)]

# Pivot the data for plotting
pivot_data = filtered_data.pivot(index='Year', columns='Item Description', values='Sale (Dollars)')

# Plotting the trend lines
plt.figure(figsize=(15, 8))
markers = ['o', 's', 'v', '^', 'D', 'p', '*', 'h', 'x', '+']  
for column, marker in zip(pivot_data.columns, markers):
    plt.plot(pivot_data.index, pivot_data[column], marker=marker, linewidth=2, label=column) 

plt.axhline(0, color='black', linestyle='--')  # Add a zero line to the y-axis

plt.title('Annual Sales Trend of Items Ranked 72nd to 81st')
plt.xlabel('Year')
plt.ylabel('Total Sales ($)')
plt.xticks(pivot_data.index)  # Set the x-axis ticks to be the years
plt.legend(title='Item Description', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
plt.grid(True)
plt.show()                                                                                                                                                                                        import matplotlib.pyplot as plt

# Identify items ranked 82nd to 91st based on total sales
items_82_to_91 = sales_sum.groupby('Item Description')['Sale (Dollars)'].sum().nlargest(91).index[81:]

# Filter the data to keep only items ranked 82nd to 91st
filtered_data = sales_sum[sales_sum['Item Description'].isin(items_82_to_91)]

# Pivot the data for plotting
pivot_data = filtered_data.pivot(index='Year', columns='Item Description', values='Sale (Dollars)')

# Plotting the trend lines
plt.figure(figsize=(15, 8))
markers = ['o', 's', 'v', '^', 'D', 'p', '*', 'h', 'x', '+']  # Adjust markers as needed
for column, marker in zip(pivot_data.columns, markers):
    plt.plot(pivot_data.index, pivot_data[column], marker=marker, linewidth=2, label=column) 

plt.axhline(0, color='black', linestyle='--')  # Add a zero line to the y-axis

plt.title('Annual Sales Trend of Items Ranked 82nd to 91st')
plt.xlabel('Year')
plt.ylabel('Total Sales ($)')
plt.xticks(pivot_data.index)  # Set the x-axis ticks to be the years
plt.legend(title='Item Description', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
plt.grid(True)
plt.show()                                                                                                                                                                                         import matplotlib.pyplot as plt

# Identify items ranked 92nd to 100th based on total sales
items_92_to_100 = sales_sum.groupby('Item Description')['Sale (Dollars)'].sum().nlargest(100).index[91:]

# Filter the data to keep only items ranked 92nd to 100th
filtered_data = sales_sum[sales_sum['Item Description'].isin(items_92_to_100)]

# Pivot the data for plotting
pivot_data = filtered_data.pivot(index='Year', columns='Item Description', values='Sale (Dollars)')

# Plotting the trend lines
plt.figure(figsize=(15, 8))
markers = ['o', 's', 'v', '^', 'D', 'p', '*', 'h', 'x', '+']  # Adjust markers as needed, add more if necessary
for column, marker in zip(pivot_data.columns, markers):
    plt.plot(pivot_data.index, pivot_data[column], marker=marker, linewidth=2, label=column) 

plt.axhline(0, color='black', linestyle='--')  # Add a zero line to the y-axis

plt.title('Annual Sales Trend of Items Ranked 92nd to 100th')
plt.xlabel('Year')
plt.ylabel('Total Sales ($)')
plt.xticks(pivot_data.index)  # Set the x-axis ticks to be the years
plt.legend(title='Item Description', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
plt.grid(True)
plt.show()                                                                                                                                                                                                  import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Aggregate sales by 'Year'
sales_by_year = df.groupby('Year')['Sale (Dollars)'].sum().reset_index()

# Create and train the linear regression model
X = sales_by_year['Year'].values.reshape(-1, 1)
y = sales_by_year['Sale (Dollars)'].values
model = LinearRegression()
model.fit(X, y)

# Predict sales for the next 8 years
next_8_years = np.array([year for year in range(X[-1, 0] + 1, X[-1, 0] + 9)]).reshape(-1, 1) 
predicted_sales = model.predict(next_8_years)

plt.figure(figsize=(14, 7))

# Ensuring that 'Year' values are integers for plotting
sales_by_year['Year'] = sales_by_year['Year'].astype(int)
next_8_years_int = next_8_years.astype(int)

# Plotting actual sales
sns.lineplot(data=sales_by_year, x='Year', y='Sale (Dollars)', marker='o', label='Actual Sales', color='blue')

# Plotting predicted sales
sns.lineplot(x=next_8_years_int.squeeze(), y=predicted_sales, label='Predicted Sales', color='red', linestyle='--')

# Title and labels with increased font size
plt.title('Total Top 100 Sales Trend and Forecast (2021 - 2028)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Total Sales (Dollars)', fontsize=14)

# Formatting the y-axis to display values in a more readable format
plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

# Adding a grid for better readability
plt.grid(True)

# Annotations to highlight the forecast start
plt.axvline(x=sales_by_year['Year'].iloc[-1], color='gray', linestyle='--', linewidth=1)
plt.text(sales_by_year['Year'].iloc[-1] + 0.5, max(y), 'Forecast Start', color='gray', fontsize=12)

# Set x-ticks to show each year as an integer
plt.xticks(sales_by_year['Year'].tolist() + next_8_years_int.squeeze().tolist(), fontsize=12)

# Increase tick label size for y-axis
plt.yticks(fontsize=12)

# Adding a legend with a specified location
plt.legend(loc='upper left', fontsize=12)

plt.tight_layout()
plt.show()


