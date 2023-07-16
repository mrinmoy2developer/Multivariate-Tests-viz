import pandas as pd
from plotly import express as px,graph_objs as go,colors
from dash import Dash, html, dcc, callback, Output, Input
import dash,os
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter

def gaussian_kernel(kernel_size,sigma):
    # Create a 2D Gaussian filter
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, kernel_size // 2] = 1
    gaussian_kernel = gaussian_filter(kernel, sigma, mode='constant')
    # Normalize the kernel
    gaussian_kernel /= np.sum(gaussian_kernel)
    return gaussian_kernel

def smoothen(Z,smoothness=1,k=5,rep=5):
    # Define a larger smoothing kernel
    kernel = gaussian_kernel(k,smoothness)
    # Apply multiple iterations of 2D convolution
    for _ in range(rep):
        Z = convolve2d(Z, kernel, mode='same', boundary='symm') / np.sum(kernel)
    # Plot the original and smoothed 3D surfaces
    return Z

app = dash.Dash(__name__)
server = app.server

directory = '../data/ALL CSV FILES'
csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]
dataframes = [pd.read_csv(os.path.join(directory, file),header=0, index_col=0) for file in csv_files]
# for j in range(15,17):
#     dataframes[j].index=[f'Dim {i}' for i in dataframes[j].index]
#     dataframes[j].columns=[f'del={float(i)/dataframes[j].shape[1]}' for i in dataframes[j].columns]
#     dataframes[j]/=1000
# cols=list(colors.PLOTLY_SCALES.keys())
cols=['Viridis','Plasma','Electric','Rainbow','Picnic','Blues','Portland','Hot','Blackbody','Jet','Greys','YlGnBu','Bluered','RdBu','YlOrRd']
marks=['#106644','purple','#ebb644','red','pink','blue','cyan','magenta','grey','yellow','brown','green','fawn','black','orange','neno']
labs=[i.split('.')[0] for i in csv_files]
# Z=[pd.read_csv(dir+csvs[i],header=0,index_col=0).values for i in range(len(csvs))]
# plots=[go.Surface(z=Z[i],colorscale=cols[i],hovertext=labs[i]) for i in range(len(csvs))]
# Define the layout
app.layout = html.Div([
    html.H1('Non-Parametric Multivariate Tests',style={'background-color': '#f2f2f2','text-align':'center'}),
    dcc.Checklist(
        id='plot-checklist',
        options=[{'label': labs[idx], 'value': idx} for idx, file in enumerate(csv_files)],
        value=[],
        labelStyle={'display': 'block','background-color': '#f2f2f2'}
    ),
    html.H3('Select the kernel Size:-',style={'background-color': '#f2f2f2','text-align':'center'}),
    dcc.Slider(
        id='Kernel-Size-slider',
        min=1,
        max=100,
        step=1,
        value=7,
        marks={i*10:f'{i*10}x{i*10}' for i in range(11)}
    ),
    html.H3('Select the Dispersion parameter of Gaussian Kernel:-',style={'background-color': '#f2f2f2','text-align':'center'}),
    dcc.Slider(
        id='Dispersion-slider',
        min=0,
        max=10,
        value=0.1,
        marks={i:f'{i}' for i in range(11)}
    ),
    html.H3('Select the Number of Iterations-',style={'background-color': '#f2f2f2','text-align':'center'}),
    dcc.Slider(
        id='num-iteration-slider',
        min=0,
        max=10,
        step=1,
        value=0,
        marks={i:f'{i}' for i in range(12)}
    ),
    dcc.Graph(
        id='surface-plot',
        figure=go.Figure(
            data=[go.Surface(z=np.zeros((10, 10)))],
            layout=go.Layout(
                title='<span style="text-align: right;">3D Surface Plots</span>',
                scene=dict(
                    xaxis=dict(title='Delta/Sigma'),
                    yaxis=dict(title='Dimension(p)'),
                    zaxis=dict(title='#Rejections')
                ),
                legend=dict(
                    orientation='v',
                    x=0,
                    y=1.1
                )
            )
        ),
        style={'height': '800px'}
    )
])

@app.callback(
    dash.dependencies.Output('surface-plot', 'figure'),
    [dash.dependencies.Input('plot-checklist', 'value'),
     dash.dependencies.Input('Kernel-Size-slider', 'value'),
     dash.dependencies.Input('Dispersion-slider', 'value'),
     dash.dependencies.Input('num-iteration-slider', 'value')]
)
def update_surface(selected_plots,k,std,rep):
    data = []
    for i in range(len(selected_plots)):
        plot_id=selected_plots[i]
        df = dataframes[plot_id]
        X=[float(i.split('=')[1]) for i in df.columns]
        Y=[float(i.split(' ')[1]) for i in df.index]
        # X, Y = np.meshgrid(df.columns, df.index)
        # Z = df.values / smoothness
        # new_data.append(go.Surface())
        z=smoothen(df.values,k=int(k),rep=rep,smoothness=std)
        trace = go.Surface(x=X,y=Y,z=z,colorscale=cols[i],hovertext=labs[plot_id])
        leg=go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode='markers',
                marker=dict(
                    color=marks[i],
                    size=20,
                    symbol='square'
                ),
                name=labs[plot_id]
            )
        data.append(trace)
        data.append(leg)

    fig = go.Figure(data=data,
                    layout=go.Layout(
            # title='Non-Parametric Multivariate Tests',
            scene=dict(
                xaxis=dict(title='Delta/Sigma'),
                yaxis=dict(title='Dimension(p)'),
                zaxis=dict(title='#Rejections')
            ),
            legend=dict(
                    orientation='v',
                    x=0,
                    y=1.1
                )
        ))
    fig.update_layout(title='<span style="text-align: right;">3D Surface Plots</span>')
    return fig

if __name__ == '__main__':
    app.run_server(debug=1)