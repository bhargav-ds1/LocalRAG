from unstructured.ingest.connector.notion.connector import NotionAccessConfig, SimpleNotionConfig
from unstructured.ingest.interfaces import PartitionConfig, ProcessorConfig, ReadConfig
from unstructured.ingest.runner import NotionRunner
# from dotenv import load_dotenv
import argparse


class NotionDataReader:
    def __init__(self, notion_api_key: str, output_dir: str = "./Data/notion-ingest-output", num_processes: int = 2,
                 verbose: bool = True, re_download: bool = False, recursive: bool = True, page_ids: list = None,
                 database_ids: list = None):
        self.processorConfig = ProcessorConfig(
            verbose=verbose,
            output_dir=output_dir,
            num_processes=num_processes,
        ),
        self.readConfig = ReadConfig(re_download=re_download)
        self.partitionConfig = PartitionConfig()
        # env = load_dotenv()
        # notion_api_key = env['notion_api_key'] if 'notion_api_key' in env.keys().tolist() else None
        self.simpleNotionConfig = SimpleNotionConfig(
            access_config=NotionAccessConfig(
                notion_api_key=notion_api_key,
            ),
            page_ids=page_ids,
            database_ids=database_ids,
            recursive=recursive,
        ),
        self.runner = NotionRunner(
            processor_config=ProcessorConfig(
                verbose=verbose,
                output_dir=output_dir,
                num_processes=num_processes,
            ),
            read_config=ReadConfig(re_download=re_download),
            partition_config=PartitionConfig(),
            connector_config=SimpleNotionConfig(
                access_config=NotionAccessConfig(
                    notion_api_key=notion_api_key if notion_api_key else None,
                ),
                page_ids=page_ids,
                recursive=recursive,
            )
        )





def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--notion_api_key", type=str, help='Provide the api key which can be '
                                                                        'used to access the related pages provided as '
                                                                        'page_ids')
    parser.add_argument("--output_dir", type=str, default="../Data/notion-ingest-output", help='Provide'
                                                                                                ' the output directory'
                                                                                                ' to place the processed'
                                                                                                ' data.')
    parser.add_argument("--num_processes", type=int, default=2)
    parser.add_argument("--verbose", action='store_true', default=False)
    parser.add_argument("--re_download",action='store_true', default=False)
    parser.add_argument("--recursive", action='store_true', default=False)
    parser.add_argument("--page_ids",   nargs='+', default=None)
    parser.add_argument("--database_ids",   nargs='+', default=None)
    args = parser.parse_args()
    # fmt: on
    return args


if __name__ == '__main__':
    args = parse_args()
    notionDataReader = NotionDataReader(args.notion_api_key,args.output_dir,args.num_processes,args.verbose,
                                        args.re_download,args.recursive,args.page_ids,args.database_ids)
    runner = notionDataReader.runner
    runner.run()
