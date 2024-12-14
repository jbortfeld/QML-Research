import pandas as pd
import time
import datetime
import psycopg2
from sqlalchemy import create_engine



def psycopg2_connect_to_rds(credentials):

    ''' connect to AWS RDS postgres database
    *** note, your IP address must be added to the allowed list in AWS RDS)
    '''
    
    conn = psycopg2.connect(
        host=credentials['host'],
        database=credentials['dbname'],
        user=credentials['user'],
        password=credentials['password'],
        port=credentials['port']
    )

    return conn

def sqlalchemy_connect_to_rds(credentials):

    engine = create_engine(
        f"postgresql+psycopg2://{credentials['user']}:{credentials['password']}@{credentials['host']}:{credentials['port']}/{credentials['dbname']}"
    )

    return engine

def upload_csv_to_table_in_rds(csv_path, table_name, engine):

    # BULK INSERT USING COPY SHOULD BE FASTER
    # df.to_csv('fundamental_dataset.csv', index=False)

    # Define connection string
    # suggested connection string:
    # engine = create_engine(
    #     "postgresql+psycopg2://postgres:quebec__ML1a@modeling-dataset.ci6paxfsercw.us-east-1.rds.amazonaws.com:5432/postgres"
    # )

    print('start upload_csv_to_table_in_rds')


    # 0. collect the column headers
    print(csv_path)
    df = pd.read_csv(csv_path, nrows=100)
    df.head(0).to_sql(table_name, engine, if_exists='replace', index=False)
    print('--table name: ', table_name)
    print('--table row count: ', f'{df.shape[0]:,}')
    print('--table column count: ', f'{df.shape[1]:,}')
    print("--table schema created successfully!")

    # 1. upload the data
    start = time.time()
    with engine.connect() as connection:
        with open(csv_path, 'r') as f:
            query = f"COPY {table_name} FROM STDIN WITH CSV HEADER"
            print('--query:',query)
            connection.connection.cursor().copy_expert(query, f)
        connection.commit()
    
    print('--table uploaded successfully in {}m'.format((time.time() - start) / 60))
    print()

    return True

def create_index_on_rds(table_name, index_name, columns_to_index, conn):

    print('start create_index_on_rds')
    print('--table name: ', table_name)
    print('--index name: ', index_name)
    print('--column name: ', columns_to_index)

    cur = conn.cursor()

    # Execute the CREATE INDEX command
    query = f"CREATE INDEX {index_name} ON {table_name} ({','.join(columns_to_index)});"
    print('--query: ', query)

    start = time.time()
    cur.execute(query)
    conn.commit()

    # Close the connection
    cur.close()
    conn.close()

    print("-- index created successfully in {}m".format((time.time() - start) / 60))
    print()

    return True