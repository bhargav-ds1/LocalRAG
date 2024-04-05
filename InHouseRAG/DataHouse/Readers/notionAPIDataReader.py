"""Notion reader."""

import os
from typing import Any, Dict, List, Optional

import requests  # type: ignore
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document

INTEGRATION_TOKEN_NAME = "NOTION_INTEGRATION_TOKEN"
BLOCK_CHILD_URL_TMPL = "https://api.notion.com/v1/blocks/{block_id}/children"
DATABASE_URL_TMPL = "https://api.notion.com/v1/databases/{database_id}/query"
SEARCH_URL = "https://api.notion.com/v1/search"


class NotionAPIPageReader():
    """Notion Page reader.

    Reads a set of Notion pages.

    Args:
        integration_token (str): Notion integration token.

    """

    is_remote: bool = True
    token: str
    headers: Dict[str, str]

    def __init__(self, integration_token: Optional[str] = None) -> None:
        """Initialize with parameters."""
        if integration_token is None:
            # integration_token = os.getenv(INTEGRATION_TOKEN_NAME)
            raise ValueError(
                "Must specify `integration_token` or set environment "
                "variable `NOTION_INTEGRATION_TOKEN`."
            )

        self.token = integration_token
        self.headers = {
            "Authorization": "Bearer " + self.token,
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28",
        }
        self.failed_database_ids = []
        self.failed_page_ids = []
        self.docs = []
        self.page_text = []
        self.json_data = []

        #super().__init__(token=token, headers=headers, failed_database_ids=failed_database_ids,
        #                 failed_page_ids=failed_page_ids)

    @classmethod
    def class_name(cls) -> str:
        """Get the name identifier of the class."""
        return "NotionPageReader"

    def _read_block(self, page_id, block_id: str, num_tabs: int = 0, recursive: bool = True,parent_page_id=None) -> str:
        """Read a block."""
        done = False
        result_lines_arr = []
        cur_block_id = block_id
        while not done:
            block_url = BLOCK_CHILD_URL_TMPL.format(block_id=cur_block_id)
            query_dict: Dict[str, Any] = {}

            res = requests.request(
                "GET", block_url, headers=self.headers, json=query_dict
            )
            data = res.json()
            data['page_id'] = page_id
            data['parent_page_id'] = parent_page_id
            self.json_data.append(data)
            for result in data["results"]:
                result_type = result["type"]
                if recursive and result_type == 'child_database':
                    self.load_data(database_id=result['id'],recursive=recursive,parent_page_id=page_id)

                result_obj = result[result_type]

                cur_result_text_arr = []
                if "rich_text" in result_obj:
                    for rich_text in result_obj["rich_text"]:
                        # skip if doesn't have text object
                        if "text" in rich_text:
                            text = rich_text["text"]["content"]
                            prefix = "\t" * num_tabs
                            cur_result_text_arr.append(prefix + text)

                result_block_id = result["id"]
                has_children = result["has_children"]
                if has_children:
                    children_text = self._read_block(
                        page_id, result_block_id, num_tabs=num_tabs + 1
                    )
                    cur_result_text_arr.append(children_text)

                cur_result_text = "\n".join(cur_result_text_arr)
                self.page_text.append(cur_result_text)
                result_lines_arr.append(cur_result_text)

            if data["next_cursor"] is None:
                done = True
                break
            else:
                cur_block_id = data["next_cursor"]

        return "\n".join(result_lines_arr)

    def read_page(self, page_id: str, recursive: bool,parent_page_id=None) -> str:
        """Read a page."""
        return self._read_block(page_id=page_id, block_id=page_id, recursive=recursive,parent_page_id=parent_page_id)

    def query_database(
            self, database_id: str, query_dict: Dict[str, Any] = {"page_size": 100}
    ) -> List[str]:
        """Get all the pages from a Notion database."""
        pages = []

        res = requests.post(
            DATABASE_URL_TMPL.format(database_id=database_id),
            headers=self.headers,
            json=query_dict,
        )
        if res.status_code != 200:
            self.failed_database_ids.append(database_id)
        try:
            res.raise_for_status()
        except Exception as e:
            print(str(e) + ':database_id:' + str(database_id))
            return []

        data = res.json()

        pages.extend(data.get("results"))

        while data.get("has_more"):
            query_dict["start_cursor"] = data.get("next_cursor")
            res = requests.post(
                DATABASE_URL_TMPL.format(database_id=database_id),
                headers=self.headers,
                json=query_dict,
            )
            res.raise_for_status()
            data = res.json()
            pages.extend(data.get("results"))

        return [page["id"] for page in pages]

    def search(self, query: str) -> List[str]:
        """Search Notion page given a text query."""
        done = False
        next_cursor: Optional[str] = None
        page_ids = []
        while not done:
            query_dict = {
                "query": query,
            }
            if next_cursor is not None:
                query_dict["start_cursor"] = next_cursor
            res = requests.post(SEARCH_URL, headers=self.headers, json=query_dict)
            data = res.json()
            for result in data["results"]:
                page_id = result["id"]
                page_ids.append(page_id)

            if data["next_cursor"] is None:
                done = True
                break
            else:
                next_cursor = data["next_cursor"]
        return page_ids

    def load_data(
            self, page_ids: List[str] = [], database_id: Optional[str] = None, recursive: bool = True,parent_page_id = None
    ) -> List[Document]:
        """Load data from the input directory.

        Args:
            page_ids (List[str]): List of page ids to load.
            database_id (str): Database_id from which to load page ids.

        Returns:
            List[Document]: List of documents.
            :param page_ids:
            :param database_id:
            :param recursive: recursively fetch inner databases of a page.

        """
        if not page_ids and not database_id:
            raise ValueError("Must specify either `page_ids` or `database_id`.")
        if database_id is not None:
            # get all the pages in the database
            page_ids = self.query_database(database_id)
            for page_id in page_ids:
                page_text = self.read_page(page_id, recursive,parent_page_id)

                self.docs.append(
                    Document(
                        text=page_text, id_=page_id, extra_info={"page_id": page_id,"parent_page_id":parent_page_id}
                    )
                )
        else:
            for page_id in page_ids:
                page_text = self.read_page(page_id=page_id, recursive=recursive,parent_page_id=parent_page_id)
                self.docs.append(
                    Document(
                        text=page_text, id_=page_id, extra_info={"page_id": page_id,"parent_page_id":parent_page_id}
                    )
                )

        return self.docs


if __name__ == "__main__":
    reader = NotionAPIPageReader()
    print(reader.search("What I"))
