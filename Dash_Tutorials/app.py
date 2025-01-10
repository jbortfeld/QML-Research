from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

def serve_layout():
    return html.Div([

        # headers that demonstrate Headers 1-5
        html.H1('Anneli Tutorial (Header 1)'),

        html.H2('-- Instructions in Dash (Header 2)'),

        html.H3('-- sample text (Header 3)'),

        html.H4('-- sample text (Header 4)'),

        html.H5('-- sample text (Header 5)'),

        # line breaks plus horizontal line
        html.Br(),
        html.Hr(),
        html.Br(),

        # paragraph
        
        html.P('This is a paragraph. It is a block of text that is displayed in a single line.'),
        html.P('Hello hello hello.'),

        html.Div('''This is a second paragraph of text. It is a block of text that is displayed in a single line.'''),

        # bold
        html.B('This is bold text.'),

        # italic
        html.I('This is italic text.'),

        # line breaks plus horizontal line
        html.Br(),
        html.Hr(),
        html.Br(),

        # rows and columns

        html.Div('Below is a row with three columns:'),

        # row
        dbc.Row([

            # row, column 1
            dbc.Col([
                html.Div('Column 1 with width 3 '),
                # html.Br(),
                html.Div('''Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna 
                         aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. 
                         Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint 
                         occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.''', style={'font-size': '20px', 'font-weight': 'bold', 'color': 'green'}),
            ], width=3, style={'border': '5px solid red'}),

            # row, column 2
            dbc.Col([
                html.Div('Column 2 with width 5 '),
                html.Br(),
                html.Div('''Justo porttitor cubilia senectus varius pellentesque laoreet himenaeos pulvinar. Lacinia quis at purus ante justo 
                         maecenas dapibus. Arcu hendrerit rutrum pellentesque mattis etiam sagittis blandit. Pretium lacus enim lacinia mauris 
                         natoque mattis donec quisque. Eros interdum semper montes donec facilisi. Facilisis nunc semper condimentum leo aliquam 
                         libero. Lobortis enim adipiscing nibh quis senectus. Massa commodo nec aenean himenaeos rhoncus dui sem velit. Aenean 
                         posuere molestie torquent; nascetur faucibus ornare habitant class. Dictum mi tempor nulla potenti tortor convallis augue. 
                         Non ipsum vestibulum pellentesque, enim sociosqu facilisi. Euismod conubia placerat nulla; id id integer. Et vehicula interdum 
                         taciti per tincidunt dis. Nec hac aenean id nulla luctus natoque.'''),
            ], width=5),

            # row, column 3
            dbc.Col([
                html.Div('Column 3 with width 4 '),
            
            ], width=4)
        ], style={'border': '1px solid blue'}),

        html.Br(),


        # add a dropdown
        html.Div([
            html.H5('this is a title for the dropdown section adfljsdfljasdfjasdfjadlkfjal;sdfjadl;sjklsdafjl;asdfjlasd;fjl;ds'),

            dcc.Dropdown(id='dropdown-input', 
                        options=[
                            {'label': 'Consumer Discretionary', 'value': 'consumer_discretionary'}, 
                            {'label': 'Option 2', 'value': 'option2'}
                            ],
                        value='consumer_discretionary',
                        style={'width': '400px'}),
        ], style={'color': 'red', 'border': '2px solid blue'}),
        


        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),







    ])

app.layout = serve_layout

if __name__ == '__main__':
    app.run(debug=True)