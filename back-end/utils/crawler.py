import os
import time
import json
import html
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException, TimeoutException, StaleElementReferenceException, UnexpectedAlertPresentException, NoAlertPresentException
from urllib.parse import urljoin
from bs4 import BeautifulSoup

class UNDocumentCrawler:
    def __init__(
        self, base_url, symbol, search_keyword, subject, pub_date_from, pub_date_to,
        search_text, language, search_type, file_extensions, jump_when_restart, download_dir
    ):
        self.base_url = base_url
        self.symbol = symbol
        self.search_keyword = search_keyword
        self.subject = subject
        self.pub_date_from = pub_date_from
        self.pub_date_to = pub_date_to
        self.search_text = search_text
        self.language = language
        self.search_type = search_type
        self.file_extensions = file_extensions
        self.jump_when_restart = jump_when_restart
        self.download_dir = download_dir
        os.makedirs(self.download_dir, exist_ok=True)

    def setup_driver(self):
        """Set up Selenium WebDriver."""
        options = Options()
        options.add_argument("--headless")  # Uncomment to run in headless mode
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        service = Service("../chromedriver-linux64/chromedriver")
        driver = webdriver.Chrome(service=service, options=options)
        return driver
    
    def find_attributes(self, block, attr):
        
        element = block.find("label", string=attr)
        if element and element.find_next("label"):
            attribute = element.find_next("label").text.strip()
            return attribute
            
        return ""  

    def find_subject(self, block, class_, label_text):
       
        container = block.find("div", class_=class_)
        if container:
            label = container.find("label", string=lambda s: s and s.strip() == label_text)
            if label:
                sibling = label.find_next_sibling("label")
                if sibling:
                    return sibling.text.strip()
        return ""  
    
    def find_publication_date(self, block, attr):
       if block.find("label", string=attr):
           pub_date = block.find("label", string=attr).find_next("label").text.strip()
           return pub_date
       return ""
    
    def crawl_page(self, html_file):
        self.download_file(html_file)

    def scroll_to_pagination(self, driver, timeout=10, max_retries=3, poll_frequency=0.5):
        retries = 0
        while retries < max_retries:
            try:
                # Wait for the element to become visible within the timeout
                pagination = WebDriverWait(driver, timeout, poll_frequency).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "lower-pagination"))
                )
                # Scroll to the pagination element
                driver.execute_script("arguments[0].scrollIntoView(true);", pagination)
                time.sleep(1)
                return  # Exit if successful
            except (NoSuchElementException, TimeoutException, StaleElementReferenceException, UnexpectedAlertPresentException) as e:
                print(f"Retrying due to {type(e).__name__}. Attempt {retries + 1} of {max_retries}.")
                retries += 1
                time.sleep(10)  # Wait before retrying
        print("Failed to locate the pagination element after maximum retries.")

    def get_total_results(self, html_file):
        try:
            div = html_file.find("div", class_="search-criteria")
            if div:
                span = div.find("span")
                if span and "Displaying results" in span.get_text():
                    total_results = span.find_all("b")[2].text.strip()
                    return total_results  # Convert to integer
        except (IndexError, ValueError, AttributeError) as e:
            print(f"An error occurred: {e}")
            pass

        return None

    def download_file(self, html_file):
        lang_map = {"Arabic": "ar", "English": "en", "French": "fr", "Chinese": "zh", "German": "de", "Russian": "ru", "Spanish": "es"}

       # Find all symbol blocks containing download links and metadata
        symbol_blocks = html_file.find_all("div", class_="symbol")
        
        for block in symbol_blocks:
            try:
                # Extract the download URL for the default document (English by default)
                download_link = block.find("a", class_="icofont-ui-file")["href"]
                download_url = html.unescape(download_link)

                # Extract the document symbol (e.g., S/RES/686(1991))
                symbol = block.find("h2", class_="h2-header-title").text.strip()

                # Filter out RESUMPTION documents
                if "RESUMPTION" in symbol or "RESUME" in symbol:
                    continue

                if self.jump_when_restart and os.path.exists('../cralwed_data/UNODS/' + symbol.replace('/', '-').replace(' ', '-')):
                    continue
    
                # Extract the title of the document
                title = block.find_next("h2").find_next("h2").text.strip() 
    
                # Find the metadata block associated with this symbol block
                metadata_block = block.find_next("div", class_="result-grid-view")

                publication_date_block = block.find_next("div", class_="lower-bar").find_next("div", class_="more-details")                
    
                # Extract metadata details
                session_year = self.find_attributes(metadata_block, "Session / Year")                
                agenda_items = self.find_attributes(metadata_block, "Agenda Item(s)")
                distribution = self.find_attributes(metadata_block, "Distribution")
                area = self.find_attributes(metadata_block, "Area")
                subjects = self.find_subject(metadata_block, "subjects", "Subject(s)")
                publication_date = self.find_publication_date(publication_date_block, "Publication Date")
    
                # Collect all available languages and their corresponding download URLs
                language_cards = metadata_block.find_next("div", class_="downloads-panel").find_all("div", class_="language-card")
                languages = []
                for card in language_cards:
                    language_name = lang_map[card.find("h4").text.strip()]
                    download_icon = card.find("i", class_="bx bxs-file-pdf")
                    if download_icon:
                        lang_url = download_url+f"&l={language_name}"
                        languages.append({"language": language_name, "url": lang_url})
    
                # Store metadata for this block
                metadata = {
                    "symbol": symbol,
                    "title": title,
                    "session_year": session_year,
                    "agenda_items": agenda_items,
                    "distribution": distribution,
                    "area": area,
                    "subjects": subjects,
                    "publication_date": publication_date,
                    "languages": [lang["language"] for lang in languages],
                }

                if not os.path.exists(self.download_dir + symbol.replace('/', '-').replace(' ', '-')):
                    os.makedirs(self.download_dir + symbol.replace('/', '-').replace(' ', '-'))

                # Download documents for each available language
                for lang in languages:
                    lang_name = lang["language"]
                    lang_url = lang["url"]
    
                    # Create a language-specific file name
                    file_name = f"{symbol.replace('/', '-').replace(' ', '-')}-{lang_name}.pdf"
                    file_path = os.path.join(self.download_dir + symbol.replace('/', '-').replace(' ', '-'), file_name)    

                    try:

                        response = requests.get(lang_url, stream=True)
                        response.raise_for_status()
        
                        with open(file_path, "wb") as file:
                            for chunk in response.iter_content(chunk_size=1024):
                                if chunk:
                                    file.write(chunk)
                    except Exception as e:
                        print(f"Error during document download: {type(e).__name__} - {e}")



                print(f"All doucments downloaded for: {symbol}")
    
                # Save metadata to JSON file
                metadata_path = os.path.join(self.download_dir + symbol.replace('/', '-').replace(' ', '-'), "metadata.jsonl")
                with open(metadata_path, "w", encoding="utf-8") as json_file:
                    json_file.write(json.dumps(metadata)+ "\n")
    
            except Exception as e:
                print(f"Error processing block: {e}")
    
    def crawl_documents(self):

        driver = self.setup_driver()
        try:
            driver.get(self.base_url)
            actions = ActionChains(driver)
            driver.set_window_size(1920, 1080)
            wait = WebDriverWait(driver, 10)

            # Input symbol
            symbol_input = wait.until(EC.presence_of_element_located((By.ID, "symbol")))
            symbol_input.send_keys(self.symbol)
            print(f"Symbol used for search: {symbol_input.get_attribute('value')}")
            time.sleep(1)

            # Input search keyword in title
            if self.search_keyword:
                title_input = wait.until(EC.presence_of_element_located((By.ID, "title")))
                title_input.send_keys(self.search_keyword)
                print(f"Keyword in title used for search: {title_input.get_attribute('value')}")
                time.sleep(1)

            # Hit subject search button
            if self.subject:
                subject_search = driver.find_element(By.ID, "bttnSubjects")
                actions.move_to_element(subject_search).click().perform()

                subject_menu_input = wait.until(EC.presence_of_element_located((By.XPATH, "//input[@placeholder='Search']")))
                subject_menu_input.send_keys(self.subject)

                subject_tag = wait.until(EC.element_to_be_clickable((By.XPATH, "//li[text()='RESOLUTIONS AND DECISIONS']")))
                driver.execute_script("arguments[0].scrollIntoView(true);", subject_tag)
                actions.move_to_element(subject_tag).double_click().perform()

                subject_ok = wait.until(EC.visibility_of_element_located((By.XPATH, "//button[text()='OK']")))
                driver.execute_script("arguments[0].scrollIntoView(true);", subject_ok)
                actions.move_to_element(subject_ok).click().perform()

                print(f"Subject used for search: {subject_menu_input.get_attribute('value')}")
                time.sleep(1)

            # Set publication date
            publication_date_from = wait.until(EC.presence_of_element_located((By.ID, "txtPublicationDateFrom")))
            publication_date_from.send_keys(self.pub_date_from)

            publication_date_to = wait.until(EC.presence_of_element_located((By.ID, "txtPublicationDateTo")))
            publication_date_to.send_keys(self.pub_date_to)

            print(f"Publication date used for search: From {publication_date_from.get_attribute('value')} to {publication_date_to.get_attribute('value')}")
            time.sleep(1)

            # Set full text search
            full_text_search = wait.until(EC.presence_of_element_located((By.ID, "searchText")))
            full_text_search.send_keys(self.search_text)
            driver.execute_script("arguments[0].dispatchEvent(new Event('input'));", full_text_search)
            driver.execute_script("arguments[0].dispatchEvent(new Event('change'));", full_text_search)
            print(f"Full text keywords used for search: {full_text_search.get_attribute('value')}")
            time.sleep(1)

            # Set target language
            target_language_menu = wait.until(EC.presence_of_element_located((By.ID, "language")))
            driver.execute_script("arguments[0].scrollIntoView(true);", target_language_menu)
            target_language_menu.click()

            target_language_option = wait.until(EC.presence_of_element_located((By.XPATH, f"//option[text()='{self.language}']")))
            driver.execute_script("arguments[0].scrollIntoView(true);", target_language_option)
            target_language_option.click()
            driver.execute_script("arguments[0].dispatchEvent(new Event('change'));", target_language_menu)
            print(f"Language used for search: {target_language_menu.get_attribute('value')}")
            time.sleep(1)

            # Set search type
            search_type_menu = wait.until(EC.presence_of_element_located((By.ID, "searchTextType")))
            driver.execute_script("arguments[0].scrollIntoView(true);", search_type_menu)
            search_type_menu.click()

            search_type_option = wait.until(EC.presence_of_element_located((By.XPATH, f"//option[text()='{self.search_type}']")))
            driver.execute_script("arguments[0].scrollIntoView(true);", search_type_option)
            search_type_option.click()
            driver.execute_script("arguments[0].dispatchEvent(new Event('change'));", search_type_menu)
            print(f"Search type used for search: {search_type_menu.get_attribute('value')}")
            time.sleep(1)

            # Hit the search button
            search_button = driver.find_element(By.ID, "btnSearch")
            driver.execute_script("arguments[0].click();", search_button)
            print("Search triggered successfully!")

            # Wait for search results to load
            time.sleep(5)

            # Parse results
            soup = BeautifulSoup(driver.page_source, "html.parser")

            cur_page = 1

            total_results = self.get_total_results(soup)
            print(f"Number of RES documents found: {total_results}" + '\n')

            print("################################################")
            print(f"Crawling documents from the {cur_page} page...")
            self.crawl_page(soup)

            has_next_page = True
            while has_next_page:
                try:
                    alert = driver.switch_to.alert
                    alert.accept()
                except NoAlertPresentException:
                    pass
                cur_page += 1
                self.scroll_to_pagination(driver)
                has_next_page = self.next_page(driver)
                time.sleep(5)
                soup = BeautifulSoup(driver.page_source, "html.parser")
                print("################################################")
                print(f"Crawling documents from the {cur_page} page...")
                self.crawl_page(soup)
        
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            driver.quit() 
        
    def next_page(self, driver):
        try:
            next_button = driver.find_element(By.XPATH, "//span[@title='Navigate to next page']")
            next_button.click()
            return True
        except NoSuchElementException:
            end_page = driver.find_element(By.XPATH, "//label")
            if end_page.text.strip() == ">":
                print("End of pagintation reached")
            return False

if __name__ == "__main__":
    crawler = UNDocumentCrawler(
        base_url="https://documents.un.org/",
        search_keyword="",
        symbol="*RES",
        subject="",
        pub_date_from="01/01/1990",
        pub_date_to="31/12/2020",
        search_text="health OR faith OR religi* OR spiritual OR belief",
        language="English",
        search_type="Use boolean operators",
        file_extensions=["pdf", "doc"],
        jump_when_restart=True,
        download_dir="/path/to/your/data/"
    )
    crawler.crawl_documents()
