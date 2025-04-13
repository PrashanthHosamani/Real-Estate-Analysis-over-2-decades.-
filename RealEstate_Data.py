# Importing essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Reading the CSV file from the current working directory
df = pd.read_csv('Real_Estate_Sales_2001-2022_GL.csv')

# Display first few rows of the dataset
print("\nFirst few rows of the dataset:")
print(df.head())

# Basic information about the dataset
print("\nDataset information:")
print(df.info())

# Display basic statistical summary
print("\nBasic statistical summary:")
print(df.describe())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Data Cleaning and Preprocessing
# 1. Handle the mixed types warning
df = pd.read_csv('Real_Estate_Sales_2001-2022_GL.csv', low_memory=False)

# 2. Convert Date Recorded to datetime
df['Date Recorded'] = pd.to_datetime(df['Date Recorded'])

# 3. Create new features
df['Year'] = df['Date Recorded'].dt.year
df['Month'] = df['Date Recorded'].dt.month

# 4. Remove any rows where Sale Amount is 0 (likely errors or incomplete data)
df = df[df['Sale Amount'] > 0]

# 5. Create a price per assessment ratio
df['Price_to_Assessment_Ratio'] = df['Sale Amount'] / df['Assessed Value']

# Display the cleaned dataset info
print("\nCleaned Dataset Information:")
print(df.info())

# Basic statistics of numerical columns
print("\nBasic Statistics of Key Numerical Columns:")
print(df[['Assessed Value', 'Sale Amount', 'Sales Ratio', 'Price_to_Assessment_Ratio']].describe())

# Save the cleaned dataset
df.to_csv('cleaned_real_estate_data.csv', index=False)

# Data Loading
df = pd.read_csv('Real_Estate_Sales_2001-2022_GL.csv', low_memory=False)

# Date Processing
df['Date Recorded'] = pd.to_datetime(df['Date Recorded'])
df['Year'] = df['Date Recorded'].dt.year
df['Month'] = df['Date Recorded'].dt.month


# Handle infinite values and outliers
# 1. Replace infinite values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# 2. Remove extreme outliers (properties over $100 million)
df = df[df['Sale Amount'] <= 100000000]

# 3. Remove properties with zero assessed value
df = df[df['Assessed Value'] > 0]

# 4. Recalculate Price_to_Assessment_Ratio
df['Price_to_Assessment_Ratio'] = df['Sale Amount'] / df['Assessed Value']

# Display updated statistics
print("\nUpdated Statistics After Cleaning:")
print(df[['Sale Amount', 'Assessed Value', 'Price_to_Assessment_Ratio']].describe())

# Keep your existing data cleaning code (lines 1-81)

# Then add this visualization code:

# Set basic style parameters
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# 1. Time Series Analysis
def plot_time_trends():
    plt.figure()
    yearly_data = df.groupby('Year').agg({
        'Sale Amount': 'mean',
        'Assessed Value': 'mean'
    })
    
    plt.plot(yearly_data.index, yearly_data['Sale Amount'], 
             marker='o', label='Sale Amount', linewidth=2)
    plt.plot(yearly_data.index, yearly_data['Assessed Value'], 
             marker='s', label='Assessed Value', linewidth=2)
    
    plt.title('Average Property Values (2001-2022)')
    plt.xlabel('Year')
    plt.ylabel('Amount ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 2. Town Analysis
def plot_top_towns():
    plt.figure()
    top_towns = df.groupby('Town')['Sale Amount'].mean().sort_values(ascending=True).tail(10)
    
    plt.barh(range(len(top_towns)), top_towns)
    plt.yticks(range(len(top_towns)), top_towns.index)
    
    plt.title('Top 10 Towns by Average Sale Price')
    plt.xlabel('Average Sale Price ($)')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(top_towns):
        plt.text(v, i, f'${v:,.0f}', va='center')
    
    plt.tight_layout()
    plt.show()

# 3. Property Type Analysis
def plot_property_types():
    plt.figure()
    prop_counts = df['Property Type'].value_counts()
    
    plt.pie(prop_counts.values[:5], 
            labels=prop_counts.index[:5],
            autopct='%1.1f%%',
            explode=[0.05]*5)
    
    plt.title('Distribution of Top 5 Property Types')
    plt.axis('equal')
    plt.show()

# 4. Monthly Patterns
def plot_monthly_patterns():
    plt.figure()
    monthly_avg = df.groupby('Month')['Sale Amount'].mean()
    
    plt.bar(monthly_avg.index, monthly_avg)
    plt.title('Average Sale Amount by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Sale Amount ($)')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(monthly_avg):
        plt.text(i, v, f'${v:,.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

# Execute all visualizations
print("\nGenerating visualizations...")
plot_time_trends()
plot_top_towns()
plot_property_types()
plot_monthly_patterns()

# Print summary statistics
print("\nMARKET ANALYSIS SUMMARY")
print("-" * 50)
print(f"Total Transactions: {len(df):,}")
print(f"Average Sale Price: ${df['Sale Amount'].mean():,.2f}")
print(f"Median Sale Price: ${df['Sale Amount'].median():,.2f}")
print(f"Most Expensive Town: {df.groupby('Town')['Sale Amount'].mean().idxmax()}")
print(f"Most Active Town: {df['Town'].value_counts().index[0]}")
print(f"Peak Sales Year: {df.groupby('Year')['Sale Amount'].count().idxmax()}")

# Add this function to your existing code

def plot_scatter_analysis():
    # Create a figure with multiple scatter plots
    plt.figure(figsize=(15, 10))
    
    # 1. Sale Amount vs Assessed Value
    plt.subplot(2, 2, 1)
    # Using only a sample of data points to avoid overcrowding
    sample_size = 10000
    sample_df = df.sample(n=sample_size, random_state=42)
    
    plt.scatter(sample_df['Assessed Value'], 
                sample_df['Sale Amount'],
                alpha=0.5)
    plt.xlabel('Assessed Value ($)')
    plt.ylabel('Sale Amount ($)')
    plt.title('Sale Amount vs Assessed Value')
    plt.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = df['Sale Amount'].corr(df['Assessed Value'])
    plt.text(0.05, 0.95, f'Correlation: {correlation:.2f}', 
             transform=plt.gca().transAxes)

    # 2. Sale Amount by Year
    plt.subplot(2, 2, 2)
    plt.scatter(sample_df['Year'], 
                sample_df['Sale Amount'],
                alpha=0.5)
    plt.xlabel('Year')
    plt.ylabel('Sale Amount ($)')
    plt.title('Sale Amount Distribution Over Years')
    plt.grid(True, alpha=0.3)

    # 3. Sale Amount by Month
    plt.subplot(2, 2, 3)
    plt.scatter(sample_df['Month'], 
                sample_df['Sale Amount'],
                alpha=0.5)
    plt.xlabel('Month')
    plt.ylabel('Sale Amount ($)')
    plt.title('Sale Amount Distribution by Month')
    plt.grid(True, alpha=0.3)

    # 4. Price to Assessment Ratio Over Time
    plt.subplot(2, 2, 4)
    # Filter out extreme ratios for better visualization
    filtered_df = sample_df[sample_df['Price_to_Assessment_Ratio'] < 5]
    plt.scatter(filtered_df['Year'], 
                filtered_df['Price_to_Assessment_Ratio'],
                alpha=0.5)
    plt.xlabel('Year')
    plt.ylabel('Price to Assessment Ratio')
    plt.title('Price/Assessment Ratio Over Time')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# Add another scatter plot for detailed price analysis
def plot_price_relationships():
    plt.figure(figsize=(15, 6))
    
    # Create two subplots side by side
    # 1. Sale Amount vs Assessed Value with Trend Line
    plt.subplot(1, 2, 1)
    sample_df = df.sample(n=10000, random_state=42)
    
    # Calculate trend line
    z = np.polyfit(sample_df['Assessed Value'], sample_df['Sale Amount'], 1)
    p = np.poly1d(z)
    
    plt.scatter(sample_df['Assessed Value'], 
                sample_df['Sale Amount'],
                alpha=0.5,
                label='Properties')
    plt.plot(sample_df['Assessed Value'], 
             p(sample_df['Assessed Value']), 
             "r--", 
             label='Trend Line')
    
    plt.xlabel('Assessed Value ($)')
    plt.ylabel('Sale Amount ($)')
    plt.title('Sale Amount vs Assessed Value\nwith Trend Line')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Sale Amount Distribution by Town (Top 10 Towns)
    plt.subplot(1, 2, 2)
    top_10_towns = df.groupby('Town')['Sale Amount'].mean().nlargest(10).index
    town_data = df[df['Town'].isin(top_10_towns)]
    
    plt.boxplot([town_data[town_data['Town'] == town]['Sale Amount'] 
                 for town in top_10_towns],
                labels=top_10_towns)
    plt.xticks(rotation=45)
    plt.ylabel('Sale Amount ($)')
    plt.title('Sale Amount Distribution\nby Top 10 Towns')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# Add these lines to your execution section
print("\nGenerating scatter plot analyses...")
plot_scatter_analysis()
plot_price_relationships()

# Additional statistical insights
print("\nCORRELATION ANALYSIS")
print("-" * 50)
print(f"Correlation between Sale Amount and Assessed Value: {df['Sale Amount'].corr(df['Assessed Value']):.3f}")
print(f"Correlation between Sale Amount and Year: {df['Sale Amount'].corr(df['Year']):.3f}")