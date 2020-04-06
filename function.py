import numpy as np
import pandas as pd
import datetime
from bokeh.models import (CDSView, ColorBar, ColumnDataSource,
                          CustomJS, CustomJSFilter, 
                          GeoJSONDataSource, HoverTool,
                          LogColorMapper, Slider, LinearColorMapper,
                          DataRange1d)
from bokeh.plotting import figure
from bokeh.palettes import brewer
from bokeh.io import output_file, show, output_notebook
import geopandas as gpd
from bokeh.models.widgets import Panel, Tabs

from scipy.stats import linregress, norm

output_notebook()

df = pd.read_csv("Corona_Virus_State.csv")
# converte csv date (formatted yearmonthday: 20200401) into more useable datetime format
def construct_date(date):
    
    year = date//10000
    month = (date%10000)//100
    day = date%100
    return datetime.datetime(year,month,day)

df["date"]=df.date.apply(construct_date)
def datetime2(x):
    return np.array(x, dtype=np.datetime64)


def get_state_list(df):
    state_list = ["US Total"]
    for i in df.state.unique():
        state_list.append(i)
    return state_list


def make_dataset(df,state):
    if state == "US Total":
        dates = np.sort(df["date"].unique())


        grouped_dataframe = df.groupby("date").sum()
        
        source = ColumnDataSource(data={
            'date'      : datetime2(dates),
            'positive' : grouped_dataframe["positive"],
            'negative' : grouped_dataframe["negative"],
            'pending' : grouped_dataframe["pending"],
            'hospitalized' : grouped_dataframe["hospitalized"],
            'death' : grouped_dataframe["death"],
            'total' : grouped_dataframe["total"],
        })
    else:
        


        grouped_dataframe = df[df["state"]==state].sort_values("date").reset_index().fillna(0)
        dates = grouped_dataframe["date"]
        source = ColumnDataSource(data={
            'date'      : datetime2(dates),
            'positive' : grouped_dataframe["positive"],
            'negative' : grouped_dataframe["negative"],
            'pending' : grouped_dataframe["pending"],
            'hospitalized' : grouped_dataframe["hospitalized"],
            'death' : grouped_dataframe["death"],
            'total' : grouped_dataframe["total"],
        })
        
    return grouped_dataframe
        
def make_plot(source,title = ''):
    
    

    p = figure(plot_height=400, x_axis_type="datetime", tools="", toolbar_location=None,
            title = title,sizing_mode="scale_width")
    p.background_fill_color="#f5f5f5"
    p.grid.grid_line_color="white"
    p.xaxis.axis_label = 'Date'
    p.yaxis.axis_label = 'Count'
    p.y_range = DataRange1d(only_visible=True)
    p.axis.axis_line_color = None
    width = 50000000
    category_list = ["death","hospitalized","positive","negative","pending"]
    color_list = ('#084594', '#2171b5', '#4292c6', '#6baed6', '#9ecae1')
    p.vbar_stack(stackers=category_list,x='date',width = width,color=color_list, source=source,legend_label = category_list)
    #p.vline_stack(stackers=category_list, x='date', color=color_list, source=source,legend_label = category_list)
    p.legend.location = "top_left"
    p.legend.click_policy="hide"
    p.add_tools(HoverTool(
        tooltips=[
            ( 'date',   '@date{%F}'            ),
            ( 'death',  '@{death}' ), # use @{ } for field names with spaces
            ( 'hospitalized',  '@{hospitalized}' ),
            ( 'positive',  '@{positive}' ),
            ( 'negative',  '@{negative}' ),
            ( 'pending',  '@{pending}' ),
        ],

        formatters={
            '@date'        : 'datetime', # use 'datetime' formatter for '@date' field
            '@{death}' : 'printf',   # use 'printf' formatter for '@{adj close}' field
            '@{hospitalized}' : 'printf',
            '@{positive}' : 'printf',
            '@{negative}' : 'printf',
            '@{pending}' : 'printf',
        },

        # display a tooltip whenever the cursor is vertically in line with a glyph
        mode='mouse'
    ))

    
    return p

def construct_date(date):
    
    year = date//10000
    month = (date%10000)//100
    day = date%100
    return datetime.datetime(year,month,day)
def produce_DataFrame():
    df = pd.read_csv("Corona_Virus_State.csv")
    df["date"]=df.date.apply(construct_date)
# converte csv date (formatted yearmonthday: 20200401) into more useable datetime format



def datetime(x):
    return np.array(x, dtype=np.datetime64)


def get_state_list(df):
    state_list = ["US Total"]
    for i in df.state.unique():
        state_list.append(i)
    return state_list


def group_dataset(df,state):
    if state == "US Total":
        dates = np.sort(df["date"].unique())


        grouped_dataframe = df.groupby("date").sum()
        
        source = ColumnDataSource(data={
            'date'      : datetime(dates),
            'positive' : grouped_dataframe["positive"],
            'negative' : grouped_dataframe["negative"],
            'pending' : grouped_dataframe["pending"],
            'hospitalized' : grouped_dataframe["hospitalized"],
            'death' : grouped_dataframe["death"],
            'total' : grouped_dataframe["total"],
        })
    else:
        


        grouped_dataframe = df[df["state"]==state].sort_values("date").reset_index().fillna(0)
        dates = grouped_dataframe["date"]
        source = ColumnDataSource(data={
            'date'      : datetime(dates),
            'positive' : grouped_dataframe["positive"],
            'negative' : grouped_dataframe["negative"],
            'pending' : grouped_dataframe["pending"],
            'hospitalized' : grouped_dataframe["hospitalized"],
            'death' : grouped_dataframe["death"],
            'total' : grouped_dataframe["total"],
        })
        
    return grouped_dataframe
        
def make_plot(source,title = ''):
    
    

    p = figure(plot_height=400, x_axis_type="datetime", tools="", toolbar_location=None,
            title = title,sizing_mode="scale_width")
    p.background_fill_color="#f5f5f5"
    p.grid.grid_line_color="white"
    p.xaxis.axis_label = 'Date'
    p.yaxis.axis_label = 'Count'
    p.y_range = DataRange1d(only_visible=True)
    p.axis.axis_line_color = None
    width = 50000000
    category_list = ["death","hospitalized","positive","negative","pending"]
    color_list = ('#084594', '#2171b5', '#4292c6', '#6baed6', '#9ecae1')
    p.vbar_stack(stackers=category_list,x='date',width = width,color=color_list, source=source,legend_label = category_list)
    #p.vline_stack(stackers=category_list, x='date', color=color_list, source=source,legend_label = category_list)
    p.legend.location = "top_left"
    p.legend.click_policy="hide"
    p.add_tools(HoverTool(
        tooltips=[
            ( 'date',   '@date{%F}'            ),
            ( 'death',  '@{death}' ), # use @{ } for field names with spaces
            ( 'hospitalized',  '@{hospitalized}' ),
            ( 'positive',  '@{positive}' ),
            ( 'negative',  '@{negative}' ),
            ( 'pending',  '@{pending}' ),
        ],

        formatters={
            '@date'        : 'datetime', # use 'datetime' formatter for '@date' field
            '@{death}' : 'printf',   # use 'printf' formatter for '@{adj close}' field
            '@{hospitalized}' : 'printf',
            '@{positive}' : 'printf',
            '@{negative}' : 'printf',
            '@{pending}' : 'printf',
        },

        # display a tooltip whenever the cursor is vertically in line with a glyph
        mode='mouse'
    ))

    
    return p

def convert(name):
    us_state_abbrev = {
        'Alabama': 'AL',
        'Alaska': 'AK',
        'American Samoa': 'AS',
        'Arizona': 'AZ',
        'Arkansas': 'AR',
        'California': 'CA',
        'Colorado': 'CO',
        'Connecticut': 'CT',
        'Delaware': 'DE',
        'District of Columbia': 'DC',
        'Florida': 'FL',
        'Georgia': 'GA',
        'Guam': 'GU',
        'Hawaii': 'HI',
        'Idaho': 'ID',
        'Illinois': 'IL',
        'Indiana': 'IN',
        'Iowa': 'IA',
        'Kansas': 'KS',
        'Kentucky': 'KY',
        'Louisiana': 'LA',
        'Maine': 'ME',
        'Maryland': 'MD',
        'Massachusetts': 'MA',
        'Michigan': 'MI',
        'Minnesota': 'MN',
        'Mississippi': 'MS',
        'Missouri': 'MO',
        'Montana': 'MT',
        'Nebraska': 'NE',
        'Nevada': 'NV',
        'New Hampshire': 'NH',
        'New Jersey': 'NJ',
        'New Mexico': 'NM',
        'New York': 'NY',
        'North Carolina': 'NC',
        'North Dakota': 'ND',
        'Northern Mariana Islands':'MP',
        'Ohio': 'OH',
        'Oklahoma': 'OK',
        'Oregon': 'OR',
        'Pennsylvania': 'PA',
        'Puerto Rico': 'PR',
        'Rhode Island': 'RI',
        'South Carolina': 'SC',
        'South Dakota': 'SD',
        'Tennessee': 'TN',
        'Texas': 'TX',
        'Utah': 'UT',
        'Vermont': 'VT',
        'Virgin Islands': 'VI',
        'Virginia': 'VA',
        'Washington': 'WA',
        'West Virginia': 'WV',
        'Wisconsin': 'WI',
        'Wyoming': 'WY'
    }
    return us_state_abbrev[name]

def make_map(dataframe,p,data,log=True):
    new_data = "adja"+data
    dataframe[new_data]=dataframe[data]+1e-11
    geosource = GeoJSONDataSource(geojson = dataframe.to_json())
    # Define color palettes

    palette = brewer['YlGnBu'][9]
    palette = palette[::-1] # reverse order of colors so higher values have darker colors
    # Instantiate LinearColorMapper that linearly maps numbers in a range, into a sequence of colors.
    second_smallest = np.partition(dataframe[new_data],-2)[5]
    third_largest = np.partition(dataframe[new_data],-2)[-5]
    if log:
        color_mapper = LogColorMapper(palette = palette, low = second_smallest, high = third_largest)
    else:
        color_mapper = LinearColorMapper(palette = palette, low = second_smallest, high = third_largest)
    # Define custom tick labels for color bar.

    # Create color bar.
    color_bar = ColorBar(color_mapper = color_mapper, 
                         label_standoff = 8,
                         border_line_color = None,
                         location = (0,0), 
                         orientation = 'horizontal')
    # Create figure object.
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    # Add patch renderer to figure.
    states = p.patches('xs','ys', source = geosource,
                       fill_color = {'field' :new_data,
                                     'transform' : color_mapper},
                       line_width = 0.25, 
                       fill_alpha = 1)
    # Create hover tool
    p.add_tools(HoverTool(renderers = [states],
                          tooltips = [('State','@STATE_NAME'),
                                    (data,'@'+data)]))
    p.x_range = DataRange1d((-200,-50))
    p.add_layout(color_bar, 'below')
    tab1 = Panel(child=p, title=data)
    return tab1

def hist_df():
    population = pd.read_csv("Population.csv").set_index("State")["Population"]
    contiguous_usa = gpd.read_file('states_21basic/states.shp')
    deaths_thousand = []
    un_adj_deaths = []
    cases_thousand = []
    un_adj_cases = []
    pop_state = []
    completed = []
    total = []
    for name in contiguous_usa.STATE_NAME.unique():
        state = convert(name)
        source = make_dataset(df,state)
        pop = int(population[name].replace(',', ''))/1000
        cases_thousand.append(max(source['total']-source['negative']-source['pending'])/pop)
        deaths_thousand.append(max(source['death'])/pop)
        completed.append(max(source['total']-source['pending']))
        un_adj_cases.append(max(source['total']-source['negative']-source['pending']))
        total.append(max(source['total']))
        un_adj_deaths.append(max(source['death']))
        pop_state.append(pop)


    data = pd.DataFrame(
        {"STATE_NAME":contiguous_usa.STATE_NAME.unique(),
        "Deaths":deaths_thousand,"Positive":cases_thousand,
        "Population":pop_state,
        "unadj_cases":un_adj_cases,
        "unadj_deaths":un_adj_deaths,
        "completed":completed,
        "total":total
        })
    return data


def map_Dataset():
    contiguous_usa = gpd.read_file('states_21basic/states.shp')
    data = hist_df()
    contiguous_usa=pd.merge(contiguous_usa,data,on="STATE_NAME")
    return contiguous_usa

def makemap():
    p1 = figure(plot_height = 600, plot_width = 600, 
               toolbar_location = 'below',
               tools = "pan, wheel_zoom, box_zoom, reset")
    p2 = figure(plot_height = 600, plot_width = 600, 
               toolbar_location = 'below',
               tools = "pan, wheel_zoom, box_zoom, reset")
    df = map_Dataset()
    tabs=Tabs(tabs=[ make_map(df,p1,"Deaths"), make_map(df,p2,"Positive")])
    show(tabs)

def hist(dataframe,data,plot,bins=25):
    hista, edges = np.histogram(dataframe[data], bins=bins)

    plot.quad(top=hista, bottom=0, left=edges[:-1], right=edges[1:], line_color="white")
    return plot
def histogram():
    data=hist_df()
    p1 = figure(plot_width=300, plot_height=300,title="Positive")

    tab1 = Panel(child=hist(data,"Positive",p1), title="Positive")

    p2 = figure(plot_width=300, plot_height=300,title="Deaths")

    tab2 = Panel(child=hist(data,"Deaths",p2), title="Deaths")

    tabs = Tabs(tabs=[ tab1, tab2 ])
    show(tabs)

def plots():
    tab1 = Panel(child=make_plot(make_dataset(df,"NY")), title="New York")
    tab2 = Panel(child=make_plot(make_dataset(df,"FL")), title="Florida")
    tab3 = Panel(child=make_plot(make_dataset(df,"KS")), title="Kansas")
    tab4 = Panel(child=make_plot(make_dataset(df,"CA")), title="California")
    tabs=Tabs(tabs=[tab1, tab2, tab3, tab4])
    show(tabs)



def solve(m1,m2,std1,std2):
    a = 1/(2*std1**2) - 1/(2*std2**2)
    b = m2/(std2**2) - m1/(std1**2)
    c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
    return np.roots([a,b,c])
def get_Vals(df,a,b):
    if a==-1 or a>len(df["positive"])-4 or b>len(df["death"])-4:
        return -1,-1,-1,-1
    mua,Ia,ra,pa,stda = linregress(range(0,len(df["positive"][a:])),np.log(df["positive"][a:]))
    mub,Ib,rb,pb,stdb = linregress(range(0,len(df["death"][b:])),np.log(df["death"][b:]))
    if mua>mub:
        return mua,stda,mub,stdb
    return mub,stdb,mua,stdb
def find_first_match(a,b):
        for i in range(0,len(a)):
            for j in range(0,len(b)):
                if a[i]==b[j] and a[i]!=0:
                    return i,j
        return -1,-1
probabilities = []
def get_prob(df,state):
    LA_data = make_dataset(df,state)
    a,b = find_first_match(LA_data["positive"],LA_data["death"])
    mu1,std1,mu2,std2 = get_Vals(LA_data,a,b)
    if mu1!=-1:
        #Get point of intersect
        result = solve(mu1,mu2,std1,std2)

        #Get point on surface
        x = np.linspace(.2,.4,10000)
        #Plots integrated area
        r = result[0]

        # integrate
        area = norm.cdf(r,mu2,std2) + (1.-norm.cdf(r,mu1,std1))
        return area
    return 0
def get_ratio():
    for state in get_state_list(df):
        if not state in ('HI','NV'): # these states are excluded due to frustrating data formats
            probabilities.append(get_prob(df,state))
    count = 0 
    for i in probabilities:
        if i>0.05:
            count+=1
    print(str(round(count*100/len(get_state_list(df)),2))+"%")



def percentmap():
    p1 = figure(plot_height = 600, plot_width = 600, 
               toolbar_location = 'below',
               tools = "pan, wheel_zoom, box_zoom, reset")
    p2 = figure(plot_height = 600, plot_width = 600, 
               toolbar_location = 'below',
               tools = "pan, wheel_zoom, box_zoom, reset")
    data = map_Dataset()
    data["deaths_per_completed"] = data["unadj_deaths"]/data["completed"]
    data["cases_per_completed"] = data["unadj_cases"]/data["completed"]
    tabs=Tabs(tabs=[ make_map(data,p1,"deaths_per_completed"), make_map(data,p2,"cases_per_completed")])
    show(tabs)

def histogram2():
    data=hist_df()
    data["deaths_per_completed"] = data["unadj_deaths"]/data["completed"]
    data["cases_per_completed"] = data["unadj_cases"]/data["completed"]
    p1 = figure(plot_width=300, plot_height=300,title="deaths per completed")

    tab1 = Panel(child=hist(data,"deaths_per_completed",p1), title="deaths per completed")

    p2 = figure(plot_width=300, plot_height=300,title="cases per completed")

    tab2 = Panel(child=hist(data,"cases_per_completed",p2), title="cases per completed")

    tabs = Tabs(tabs=[ tab1, tab2 ])
    show(tabs)


def testedmap():
    p1 = figure(plot_height = 600, plot_width = 600, 
               toolbar_location = 'below',
               tools = "pan, wheel_zoom, box_zoom, reset")
    p2 = figure(plot_height = 600, plot_width = 600, 
               toolbar_location = 'below',
               tools = "pan, wheel_zoom, box_zoom, reset")
    data = map_Dataset()
    data["Evaluated"] = data["completed"]/data["Population"]
    data["Total"] = (data["total"])/(data["Population"])
    tabs=Tabs(tabs=[ make_map(data,p1,"Evaluated",log=False), make_map(data,p2,"Total",log=False)])
    show(tabs)