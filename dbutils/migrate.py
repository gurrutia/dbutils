"""Efficiently migrate data between databases using multithreaded processing and connection pooling."""
import concurrent.futures
import logging
import threading
from typing import Generator, List

import psycopg2.pool
from psycopg2.extensions import connection
from psycopg2.extras import execute_values

# Basic logging configuration, customize as needed
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s [PID%(process)d|TID%(thread)d|%(threadName)s|%(name)s|L%(lineno)d] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)


class ThreadedConnectionPoolManager:
    """Thread-safe threaded connection pool manager for PostgreSQL databases."""

    def __init__(self, connection_params: dict, minconn: int = 1, maxconn: int = 5):
        try:
            self.pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=minconn, maxconn=maxconn, **connection_params
            )
            self._lock = threading.Lock()
            self.pool_key = (
                f"{connection_params['user']}@{connection_params['host']}:"
                f"{connection_params['port']}/{connection_params['database']}"
                f" (minconn=1, maxconn={maxconn})"
            )
        except psycopg2.Error as e:
            logging.error(f"Error occurred while creating connection pool: {e}")
            raise
        finally:
            logging.info(f"Initialized connection pool ~ {self.pool_key}")

    def get_connection(self) -> connection:
        """Get a connection from the pool."""
        try:
            with self._lock:
                return self.pool.getconn()
        except psycopg2.Error as e:
            logging.error(f"Error occurred while getting connection from pool: {e}")
            raise
        finally:
            logging.info(f"Got connection from pool < {self.pool_key}")

    def release_connection(self, conn: connection) -> None:
        """Release a connection back to the pool.

        :param conn: The connection to release back to the pool.
        """
        try:
            with self._lock:
                self.pool.putconn(conn)
        except psycopg2.Error as e:
            logging.error(f"Error occurred while releasing connection to pool: {e}")
            raise
        finally:
            logging.info(f"Released connection back to pool > {self.pool_key}")

    def close_all_connections(self) -> None:
        """Close all connections in the pool."""
        try:
            with self._lock:
                self.pool.closeall()
        except psycopg2.Error as e:
            logging.error(f"Error occurred while closing all connections in pool: {e}")
            raise
        finally:
            logging.info(f"Closed all connections to {self.pool_key}")

    def __repr__(self) -> str:
        return f"ThreadedConnectionPoolManager({self.pool_key})"

    def __str__(self) -> str:
        return f"ThreadedConnectionPoolManager({self.pool_key})"


class DataProcessor:
    """Multithreaded data processor for migrating data between databases."""

    def __init__(
        self,
        source_manager: ThreadedConnectionPoolManager,
        target_manager: ThreadedConnectionPoolManager,
    ):
        self.source_manager = source_manager
        self.target_manager = target_manager

    def select_data(
        self, query: str, batch_size: int = 1000
    ) -> Generator[List[tuple], None, None]:
        """Select data from the source database.

        :param query: The SELECT query to fetch data from the source database.
        :param batch_size: The number of rows to fetch in each batch.
        :yield: A batch of data fetched from the source database.
        """
        logging.info(f"Fetching from source {self.source_manager.pool_key}")
        try:
            conn = self.source_manager.get_connection()
            with conn.cursor() as cursor:
                cursor.itersize = batch_size
                cursor.execute(query)

                # Fetch and yield data in batches
                while True:
                    batch = cursor.fetchmany(batch_size)
                    if not batch:
                        logging.info("No more data to fetch from source")
                        break
                    logging.info(f"Fetched {len(batch)} rows")
                    yield batch
        except psycopg2.Error as e:
            logging.error(f"Error occurred while selecting data: {e}")
            raise
        finally:
            self.source_manager.release_connection(conn)

    def insert_data(
        self, query: str, data: List[tuple], batch_size: int = 1000
    ) -> None:
        """Insert data into the target database.

        :param query: The INSERT query to insert data into the target database.
        :param data: The data to insert into the target database.
        :param batch_size: The number of rows to insert in each batch.
        """
        logging.info(f"Inserting into target {self.target_manager.pool_key}")

        conn = self.target_manager.get_connection()
        try:
            with conn.cursor() as cursor:
                execute_values(cursor, query, data, page_size=batch_size)
            conn.commit()
            logging.info(f"Inserted {len(data)} rows")
        except psycopg2.Error as e:
            conn.rollback()
            logging.error(f"Error occurred while inserting data: {e}")
            raise
        finally:
            self.target_manager.release_connection(conn)

    def insert_selected_data(
        self,
        select_query: str,
        insert_query: str,
        batch_size: int = 1000,
        thread_workers: int = 5,
        thread_name_prefix: str = "DataProcessorThread",
    ) -> None:
        """Insert selected data in batches using multiple threads."""
        if batch_size <= 0:
            logging.warning("Batch size must be greater than 0")
            batch_size = 1000

        logging.info(
            f"Inserting selected data in batches of {batch_size} using {thread_workers} threads"
        )

        selected_data_batches = self.select_data(select_query, batch_size)

        # Define a function to insert data using a single thread
        def insert_data_thread(data: List[tuple]) -> None:
            """Insert data using a single thread. This function is called by multiple threads."""
            try:
                self.insert_data(insert_query, data, batch_size)
            except Exception as e:
                logging.error(f"Error occurred during data insertion: {e}")

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=thread_workers,
            thread_name_prefix=thread_name_prefix,
        ) as executor:
            futures = [
                executor.submit(insert_data_thread, data)
                for data in selected_data_batches
            ]
            concurrent.futures.wait(futures)


def example() -> None:
    # Source database connection parameters
    source_params = {
        "host": "your_source_host_here",
        "port": "your_port_here",
        "database": "your_source_db",
        "user": "your_user_here",
        "password": "your_password_here",
    }

    # Target database connection parameters
    # NOTE: Your target database table should exist
    target_params = {
        "host": "your_target_host_here",
        "port": "your_port_here",
        "database": "your_target_db",
        "user": "your_user_here",
        "password": "your_password_here",
    }

    # Initialize source and target ThreadedConnectionPoolManager's
    source_manager = ThreadedConnectionPoolManager(source_params)
    target_manager = ThreadedConnectionPoolManager(target_params)

    # Initialize data processor using source and target managers
    data_processor = DataProcessor(source_manager, target_manager)

    # Select data to migrate from source to target
    # You must define the select and insert queries for your use case
    # The insert query must have a %s placeholder for the VALUES
    select_query = "SELECT col1, col2 FROM source_table;"
    insert_query = "INSERT INTO target_table (col1, col2) VALUES %s;"

    # Insert selected data in batches using multiple threads
    try:
        data_processor.insert_selected_data(select_query, insert_query)
    except Exception as e:
        logging.error(f"Error occurred during data processing: {e}")
    finally:
        source_manager.close_all_connections()
        target_manager.close_all_connections()


if __name__ == "__main__":
    example()
