import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

# Load the preprocessed data
file_path = 'covid_19_clean_complete.csv'
data = pd.read_csv(file_path)
data['Date'] = pd.to_datetime(data['Date'])
data['Province/State'] = data['Province/State'].fillna('Unknown')
aggregated_data = data.groupby(['Country/Region', 'Date']).agg(
    {
        'Confirmed': 'sum',
        'Deaths': 'sum',
        'Recovered': 'sum',
        'Active': 'sum',
        'Lat': 'mean',
        'Long': 'mean',
    }
).reset_index()

# Feature engineering for predictions
aggregated_data['Days_Since'] = (aggregated_data['Date'] - aggregated_data['Date'].min()).dt.days

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Pandemic Prediction Dashboard"

# Layout of the dashboard
app.layout = html.Div([
    # Navbar
    html.Div([
        html.H1("Pandemic Prediction Dashboard", style={
            'textAlign': 'center', 
            'padding': '10px', 
            'borderBottom': '2px solid black',
            'marginBottom': '20px'}),
    ], style={
        'backgroundColor': '#343a40', 
        'color': 'white', 
        'padding': '10px', 
        'boxShadow': '0px 4px 6px rgba(0, 0, 0, 0.1)'}),

    # Dropdown for country selection
    html.Div([
        dcc.Dropdown(
            id='country-dropdown',
            options=[{'label': country, 'value': country} for country in aggregated_data['Country/Region'].unique()],
            value='United States',
            placeholder="Select a country",
            style={'width': '50%', 'margin': '20px auto', 'padding': '10px', 'fontSize': '16px'}
        )
    ]),

    # Input for days selection for prediction
    html.Div([
        html.Label("Enter number of days for prediction:", style={'fontSize': '18px', 'marginBottom': '10px'}),
        dcc.Input(id='days-input', type='number', value=30, min=1, max=1000, step=1, 
                  style={'width': '10%', 'margin': '10px auto', 'padding': '5px', 'textAlign': 'center'}),
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),

    # Predictions table
    html.Div([
        html.H4("Predictions for Selected Country:", style={'textAlign': 'center', 'textDecoration': 'underline'}),
        dash_table.DataTable(
            id='predictions-table',
            style_table={'margin': 'auto', 'width': '80%'},
            style_header={
                'backgroundColor': '#f4f4f4',
                'fontWeight': 'bold',
                'textAlign': 'center'
            },
            style_cell={
                'textAlign': 'center',
                'padding': '10px'
            }
        )
    ], style={'marginBottom': '20px'}),

    # Spread map and time series plot
    dcc.Graph(id='time-series-plot', style={'marginBottom': '20px'}),
    dcc.Graph(id='global-map', style={'marginBottom': '20px'}),

    # Footer
    html.Div([
        html.P("Pandemic Prediction Dashboard © 2025", style={'textAlign': 'center', 'padding': '10px'}),
    ], style={
        'backgroundColor': '#343a40', 
        'color': 'white', 
        'marginTop': '20px', 
        'padding': '10px'})
])

# Callbacks for interactivity
@app.callback(
    [Output('time-series-plot', 'figure'),
     Output('global-map', 'figure'),
     Output('predictions-table', 'data'),
     Output('predictions-table', 'columns')],
    [Input('country-dropdown', 'value'),
     Input('days-input', 'value')]
)
def update_dashboard(country, days):
    if not days or days <= 0:
        days = 30

    # Filter data for the selected country
    country_data = aggregated_data[aggregated_data['Country/Region'] == country]

    # Time series plot
    fig_time_series = px.line(
        country_data,
        x='Date',
        y=['Confirmed', 'Deaths', 'Recovered', 'Active'],
        title=f"Trends in {country}",
        labels={"value": "Cases", "Date": "Date"},
        template="plotly_dark"
    )
    fig_time_series.update_layout(legend_title_text='Metrics')

    # Global map
    latest_data = aggregated_data[aggregated_data['Date'] == aggregated_data['Date'].max()]
    fig_global_map = px.scatter_geo(
        latest_data,
        lat='Lat',
        lon='Long',
        size='Confirmed',
        color='Deaths',
        hover_name='Country/Region',
        title="Global Spread of Pandemic",
        projection="natural earth",
        template="plotly_dark"
    )

    # Prepare data for predictions
    X = country_data[['Days_Since']]
    y_confirmed = country_data['Confirmed']
    y_deaths = country_data['Deaths']

    # Train models
    poly_model_confirmed = make_pipeline(PolynomialFeatures(degree=2), StandardScaler(), LinearRegression())
    poly_model_deaths = make_pipeline(PolynomialFeatures(degree=2), StandardScaler(), LinearRegression())

    if len(X) > 2:  # Ensure enough data for training
        poly_model_confirmed.fit(X, y_confirmed)
        poly_model_deaths.fit(X, y_deaths)

        future_days = np.arange(country_data['Days_Since'].max() + 1, country_data['Days_Since'].max() + days + 1).reshape(-1, 1)
        future_confirmed = poly_model_confirmed.predict(future_days)
        future_deaths = poly_model_deaths.predict(future_days)

        # Calculate the peak case load date
        peak_date = country_data['Date'].max() + pd.Timedelta(days=days)

        predictions_data = [
            {"Metric": f"Predicted Confirmed Cases ({days} Days)", "Value": int(max(future_confirmed[-1], 0))},
            {"Metric": f"Predicted Deaths ({days} Days)", "Value": int(max(future_deaths[-1], 0))},
            {"Metric": "Peak Case Load Estimate", "Value": peak_date.date()}
        ]
    else:
        predictions_data = [{"Metric": "Insufficient Data", "Value": "Predictions unavailable."}]

    predictions_columns = [{"name": "Metric", "id": "Metric"}, {"name": "Value", "id": "Value"}]

    return fig_time_series, fig_global_map, predictions_data, predictions_columns

if __name__ == '__main__':
    app.run_server(debug=True)
