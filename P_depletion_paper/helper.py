import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import pandas as pd

coarse_to_basic_mapping = {
1: [1, 2],
2: [3, 8],
3: [4, 5],
4: [6, 7],
5: [9, 10],
6: [11, 12],
7: [13],
8: [14, 15],
9: [16, 17, 18]
}

# Mapping for "simple" to "basic"
simple_to_basic_mapping = {
    0: list(range(1, 11)),
    1: list(range(11, 19))
}
sub_to_basic_mapping = {
    0: list(range(1, 19)),
}
    
def plot_3d_plane(xlabel, ylabel, zlabel, grouped, typesample='root', treatments=['0P', '100P']):
    roots = grouped[grouped['type'] == typesample]
    roots = roots[roots['plate'] != 431]
    roots = roots[roots['plate'] >= 10]
    roots = roots[roots['treatment'].isin(treatments)]

    # Extract the data
    x = roots[xlabel].values
    y = roots[ylabel].values
    z = roots[zlabel].values
    treatment = roots['treatment'].values  # Extract treatment values

    # Prepare the data for least squares fitting
    # We use a constant term of 1 to account for the D coefficient
    A = np.c_[x, y, np.ones(x.shape)]

    # Perform least squares fitting
    C, _, _, _ = np.linalg.lstsq(A, z, rcond=None)

    # The plane equation will be of the form: z = Ax + By + D
    A_fit, B_fit, D_fit = C

    # Generate points for plotting the plane
    xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), 100),
                         np.linspace(y.min(), y.max(), 100))

    # Calculate corresponding z from the plane equation
    zz = A_fit * xx + B_fit * yy + D_fit

    # Create the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define colors for different treatments
    colors = {'0P': 'cyan', '100P': 'blue'}

    # Scatter plot with different colors for treatments
    for t in treatments:
        mask = (treatment == t)
        ax.scatter(x[mask], y[mask], z[mask], color=colors[t], label=f'Data points ({t})')

    # Plot the best-fit plane
    ax.plot_surface(xx, yy, zz, color='g', alpha=0.5, label='Fitted Plane')

    # Add labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(f'{zlabel} = {A_fit:.1e} * {xlabel} + {B_fit:.1e} * {ylabel} + {D_fit:.1e}', size=12)

    # Add a legend

    return fig, ax


def plot_scatter_with_mean_and_ci(df, x_col, y_col, ax, label=None):
    n_colors = len(df['strain'].unique())
    palette = sns.color_palette("rainbow", n_colors)
    strain_to_color = dict(zip(df['strain'].unique(), palette))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue='strain', palette=strain_to_color, ax=ax)
    sns.regplot(data=df, x=x_col, y=y_col, scatter=False, color="black", ax=ax)

    for strain, color in strain_to_color.items():
        sub_df = df[df['strain'] == strain]
        x_mean = sub_df[x_col].mean()
        y_mean = sub_df[y_col].mean()
        x_ci = sub_df[x_col].sem()
        y_ci = sub_df[y_col].sem()
        ax.errorbar(x=x_mean, y=y_mean, xerr=x_ci, yerr=y_ci, color=color, fmt='o')

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    if label:
        ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=16, fontweight='bold', va='top')
    

def get_regions(arrangement, index):
    if arrangement == 'basic':
        indexes = [index-1]
    if arrangement == 'simple':
        indexes = [basic_pos-1 for basic_pos in simple_to_basic_mapping[index]]
    if arrangement == 'coarse':
        indexes = [basic_pos-1 for basic_pos in coarse_to_basic_mapping[index]]
    if arrangement == 'sub':
        indexes = [basic_pos-1 for basic_pos in sub_to_basic_mapping[index]] 
    return(indexes)

def calculate_integral(df, column, new_column):
    # Calculate the time difference within each group
    df['time_since_begin_hour'] = df['time_since_begin_h'].dt.total_seconds() / 3600.0
    df['time_diff'] = df.groupby('unique_id')['time_since_begin_hour'].transform(lambda x: x.diff())

    # Calculate the average length density within each group
    df['avg_length_density'] = df.groupby('unique_id')[column].transform(lambda x: x.rolling(window=2).mean())

    # Calculate the "area" (using Trapezoidal rule) for each pair of rows within each group
    df['area'] = df['time_diff'] * df['avg_length_density']

    # Perform the integration (cumulative sum of "area") within each group
    df[new_column] = df.groupby('unique_id')['area'].transform(lambda x: x.cumsum())

    # Drop the helper columns if needed
    df.drop(['time_diff', 'avg_length_density', 'area'], axis=1, inplace=True)
    


def zero_regplot(data, x, y, ax, label, scatter=True, n_boot=1000, color=None):
    # Extract x and y data
    x_data = data[x]
    y_data = data[y]

    # Bootstrap resampling
    bootstrapped_slopes = []
    for _ in range(n_boot):
        # Resample with replacement
        resampled_indices = np.random.choice(range(len(x_data)), size=len(x_data), replace=True)
        resampled_x = x_data.iloc[resampled_indices]
        resampled_y = y_data.iloc[resampled_indices]

        # Fit a line through the origin
        slope = np.sum(resampled_x * resampled_y) / np.sum(resampled_x * resampled_x)
        bootstrapped_slopes.append(slope)

    # Calculate slope and confidence intervals
    bootstrapped_slopes = np.array(bootstrapped_slopes)
    slope_estimate = bootstrapped_slopes.mean()
    ci_lower = np.percentile(bootstrapped_slopes, 2.5)
    ci_upper = np.percentile(bootstrapped_slopes, 97.5)

    # Create regression line
    x_vals = x_data
    y_vals = slope_estimate * x_vals

    # Plot
    if scatter:
        ax.scatter(x_data, y_data, alpha=0.7,s=3)
    ax.plot(x_vals, y_vals, color=color, label=f'{label} Slope: {slope_estimate:.2f}')
    ax.fill_between(x_vals, ci_lower * x_vals, ci_upper * x_vals, color=color, alpha=0.2)

    # Set labels and legend
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend()