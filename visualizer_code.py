# Import necessary modules
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew

# Define functions

def line_plot(data, x_label, y_label, title, labels):
    """
    Creates a line plot for multiple lines with appropriate labels and legend.

    Args:
        data (DataFrame): Data to be plotted. Each column represents a line.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        title (str): Title of the plot.
        labels (list of str): Labels for each line.

    Returns:
        None
    """
    plt.figure(figsize=(12, 7))
    for i in range(len(data.columns)):
        plt.plot(data.index, data.iloc[:, i], label=labels[i], linewidth=3, linestyle='-', marker='o')
    plt.xlabel(x_label, fontsize=14, fontweight='bold')
    plt.ylabel(y_label, fontsize=14, fontweight='bold')
    plt.title(title, fontsize=18, fontweight='bold', style='italic')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.show()


def histogram_plot(data, x_label, y_label, title):
    """
    Creates a histogram to show the distribution of the dataset.

    Args:
        data (Series): Data to be plotted.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        title (str): Title of the plot.

    Returns:
        None
    """
    plt.figure(figsize=(12, 7))
    plt.hist(data, bins=15, alpha=0.75, rwidth=0.85, color='#1f77b4')
    plt.xlabel(x_label, fontsize=14, fontweight='bold')
    plt.ylabel(y_label, fontsize=14, fontweight='bold')
    plt.title(title, fontsize=18, fontweight='bold', style='italic')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.show()

    # Print statistical summary
    print("Descriptive Statistics:")
    print(data.describe())
    print(f"Kurtosis: {kurtosis(data)}")
    print(f"Skewness: {skew(data)}")


def bar_chart(labels, data, x_label, y_label, title):
    """
    Creates a bar chart to compare different values.

    Args:
        labels (list of str): Labels for each bar.
        data (list): Data to be plotted.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        title (str): Title of the plot.

    Returns:
        None
    """
    plt.figure(figsize=(14, 8))
    colors = ['#ff7f0e', '#2ca02c', '#1f77b4', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    plt.bar(labels, data, color=colors[:len(labels)], alpha=0.9)
    plt.xlabel(x_label, fontsize=14, fontweight='bold')
    plt.ylabel(y_label, fontsize=14, fontweight='bold')
    plt.title(title, fontsize=18, fontweight='bold', style='italic')
    plt.xticks(rotation=45, fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def pie_chart(data, labels, title):
    """
    Creates a pie chart to compare proportions.

    Args:
        data (list): Data to be plotted.
        labels (list of str): Labels for each slice.
        title (str): Title of the plot.

    Returns:
        None
    """
    plt.figure(figsize=(10, 10))
    plt.pie(data, labels=labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors, wedgeprops={'linewidth': 1, 'edgecolor': 'black'})
    plt.title(title, fontsize=18, fontweight='bold', style='italic')
    plt.axis('equal')
    plt.show()


def binnify(age):
    """
    Assigns an age band based on the age value.

    Args:
        age (int): Age of the person.

    Returns:
        str: Age band category.
    """
    if age < 12:
        return 'Child (<12)'
    elif 12 < age < 20:
        return 'Teenager (12 - 20)'
    elif 20 <= age <= 30:
        return 'Young Adult (20 - 30)'
    elif 30 < age <= 40:
        return 'Adult (30 - 40)'
    elif 40 < age <= 50:
        return 'Older Adult (40 - 50)'
    elif age > 50:
        return 'Old (50+)'


# Main program

# Visualization 1: Histogram of Age Distribution of Casualties
data = pd.read_csv('dft-road-casualty-statistics-casualty-2022.csv')
histogram_plot(data['age_of_casualty'], 'Age of Casualty', 'Frequency', 'Age Distribution of Road Casualties')

# Visualization 2: Pie Charts for Age Bands of Drivers and Casualties
data = pd.read_csv('dft-road-casualty-statistics-vehicle-2022.csv')
data['age_band'] = data['age_of_driver'].apply(lambda x: binnify(x))
df = data['age_band'].value_counts()
print(df.head())
pie_chart(df.values, df.index, 'Age Bands Responsible for Accidents')

data = pd.read_csv('dft-road-casualty-statistics-casualty-2022.csv')
data['age_band'] = data['age_of_casualty'].apply(lambda x: binnify(x))
df = data['age_band'].value_counts()
pie_chart(df.values, df.index, 'Age Bands Most Affected by Accidents')

# Visualization 3: Bar Chart of Top 20 Car Models Involved in Accidents
data = pd.read_csv('dft-road-casualty-statistics-vehicle-2022.csv')
df = data['generic_make_model'].value_counts()
df = df.iloc[1:21]
bar_chart(df.index, df.values, 'Car Models', 'Number of Casualties', 'Top 20 Car Models Involved in Road Accidents')

# Visualization 4: Line Plot of Casualties Caused by Different Car Models Over the Years
data1 = pd.read_csv('dft-road-casualty-statistics-vehicle-last-5-years.csv')
data2 = pd.read_csv('dft-road-casualty-statistics-casualty-last-5-years.csv')
data = pd.merge(data1, data2, on='accident_index', how='inner')
top_n = data['generic_make_model'].value_counts().nlargest(8)
top_n_index = top_n.index[1:]
df = data[data['generic_make_model'].isin(top_n_index)]
df = pd.pivot_table(df, columns=['generic_make_model'], values=['lsoa_of_casualty'], index='accident_year_x', aggfunc='count')
df.columns = df.columns.droplevel(level=0)
df.columns.name = None
line_plot(df, 'Year', 'Number of Casualties', 'Yearly Trends of Casualties by Car Model', top_n_index)