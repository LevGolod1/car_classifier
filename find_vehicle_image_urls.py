import selenium
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from random_user_agent.user_agent import UserAgent
from random_user_agent.params import SoftwareName, OperatingSystem

import random
import time
import matplotlib.pyplot as plt
import re
import cv2
import os
import numpy as np
import pandas as pd
import csv
import traceback
from timethis import timethis


####################################
### Settings
geckodriver_path = "/usr/local/bin/geckodriver"
parent_directory_url_csvs='/Users/levgolod/Projects/car_classifier/data/autotrader/vehicle_metadata/'
parent_directory_images='/Users/levgolod/Projects/car_classifier/data/autotrader/vehicle_images/'
wait_time=10
scroll_pause_time = 0.5  # Adjust based on load time
scroll_distance = 500 # pixels
max_scrolls = 100
# headless=True
headless=False
####################################

def load_zipcodes() -> list:
    '''
    Source https://simplemaps.com/data/us-zips
    :return:
    '''
    # zips_filepath='~/Projects/car_classifier/data/simplemaps_uszips_basicv1.90/uszips.csv'
    zips_filepath='./data/simplemaps_uszips_basicv1.90/uszips.csv'
    zips_df = pd.read_csv(zips_filepath)
    zips_df['zip'] =zips_df['zip'].astype(str).str.zfill(5)
    return list(zips_df['zip'])


def clean_vehicle_url(url:str) -> str:
    pattern = r'^(https://.*?/vehicle/[a-zA-Z0-9]*)(\?.*)?$'
    re_result = re.match(pattern, str(url))
    if bool(re_result):
        return re_result.group(1)
    else:
        return None


def get_clean_vin(vin_text:str) -> str:
    '''
    'VIN: 4T1G11AK5NU020242'
    :param vin_text:
    :return:
    '''
    vin = str(vin_text).upper().split(':')[-1].strip()
    if len(vin) ==17:
        return vin
    else:
        return None


def get_search_result_count(driver) -> int:
    result_count_element = driver.find_element(By.CSS_SELECTOR, '[data-cmp="resultsCount"]')
    result_count_text = result_count_element.text
    result_count = int(result_count_text.split(' ')[0].replace(',',''))
    return result_count


def find_image_urls(driver) -> list:
    # get the image urls. There are multiple sizes available.
    # I used to 'keep it one hundred', but due to inflation I am gonna 'keep it 500'
    image_url_prefix = 'https://images.autotrader.com/scaler/500'
    return list(filter(
        lambda x: str(x).startswith(image_url_prefix),
        str(driver.page_source).split()
    ))


def clean_text_remove_newline(x:str) -> str:
    # return '|'.join(str(x).strip().replace('|',' ').split('\n'))
    special_chars = '|^'
    internal_delim = '^'
    chars_to_remove = sorted(set(special_chars + internal_delim))
    x2 = str(x).strip()
    for char in chars_to_remove:
        x2 = x2.replace(char, ' ')

    lines= [y.strip() for y in x2.strip().split('\n')]
    return internal_delim.join(lines)


def find_vehicle_listing_links(driver) -> list:
    '''
    return list of tuples
    [
         (
            'Certified 2021 Porsche Taycan 4 Cross Turismo',
            'https://www.autotrader.com/cars-for-sale/vehicle/737243275?listingType=USED&makeCode=POR&modelCode=PORTAYCAN&numRecords=25&referrer=%2Fcars-for-sale%2Fporsche%2Ftaycan&sortBy=relevance&zip=92101&clickType=spotlight'
        ),
    ]
    '''
    try:
        WebDriverWait(driver, wait_time).until(
            EC.visibility_of_element_located(
                (By.XPATH, '//a[starts-with(@href, "/cars-for-sale/vehicle/")]')
            )
        )
        elements = driver.find_elements(By.XPATH, '//a[starts-with(@href, "/cars-for-sale/vehicle/")]')

    except (NoSuchElementException, TimeoutException):
        elements =[]


    return list(map(
        lambda element: (element.text, element.get_attribute("href")),
        elements
    ))

def get_scroll_percentage(driver):
    """Returns how far down the page is scrolled as a percentage."""
    scroll_top = driver.execute_script("return window.scrollY;")  # Current scroll position
    scroll_height = driver.execute_script("return document.documentElement.scrollHeight;")  # Total scroll height
    client_height = driver.execute_script("return window.innerHeight;")  # Viewport height

    # Calculate percentage (0 to 100)
    scroll_percentage = (scroll_top / (scroll_height - client_height)) * 100
    return round(scroll_percentage, 2)


def scroll_down_incrementally(driver, scroll_distance=scroll_distance, wait=0.05, verbose=False):
    # todo should start by scrolling all the way to the top?
    current_pct = get_scroll_percentage(driver)
    i=0
    while (current_pct < 100) and (i<=max_scrolls):
        i+=1
        message = f'scrolling {i}, {current_pct}%'
        if verbose:
            print(message)
        driver.execute_script(f"window.scrollBy(0, {scroll_distance});")
        current_pct = get_scroll_percentage(driver)
        time.sleep(wait)

def create_random_user_agent():
    software_names = [SoftwareName.CHROME.value, SoftwareName.FIREFOX.value]
    operating_systems = [OperatingSystem.WINDOWS.value, OperatingSystem.MACOS.value]
    ua = UserAgent(software_names=software_names, operating_systems=operating_systems, limit=100)
    random_user_agent = ua.get_random_user_agent()
    return random_user_agent


def firefox_driver_init(headless:bool=False, randomize:bool=True) -> webdriver.Firefox:


    firefox_options = Options()

    random_user_agent = create_random_user_agent()
    firefox_options.set_preference("general.useragent.override", random_user_agent)

    if headless:
        firefox_options.add_argument("--headless")

    service = Service(geckodriver_path)
    driver = webdriver.Firefox(service=service, options=firefox_options)

    return driver

def get_image_urls_from_view_all_media_button(driver) -> list:
    '''
    Assuming there is a button "View All Media" click on it and scroll through the images

    :param driver:
    :return:
    '''
    ## 'view all media' button to see the images
    view_all_media_element = WebDriverWait(driver, wait_time).until(
        EC.visibility_of_element_located(
            (By.XPATH, "//p[normalize-space(text())='View All Media']")
        )
    )
    view_all_media_element.click()
    print("view_all_media clicked successfully")

    ## scroll through all the photos
    scroll_panel = WebDriverWait(driver, wait_time).until(
        EC.visibility_of_element_located(
            (By.CSS_SELECTOR, "div[data-cmp='modalScrollPanel']")
        )
    )

    ## try to scroll in a more naturalistic way    # scroll gradually until there all new image urls are captured
    image_urls = find_image_urls(driver)
    last_length = len(image_urls)
    i = 0
    while True:
        # Scroll down by a small amount
        driver.execute_script(f"arguments[0].scrollTop += {scroll_distance};", scroll_panel)
        # print(f'they see me scrollin, they hatin ({i})')
        i += 1
        time.sleep(scroll_pause_time)  # Wait for content to load
        image_urls = find_image_urls(driver)
        new_length = len(image_urls)
        if (new_length == last_length) or (i >= max_scrolls):  # If no more content is loading, break
            print(f'cant scroll anymore boss {i}')
            break
        last_length = new_length

    return image_urls


def process_vehicle_webpage(url:str, quit:bool=True):

    url_cleaned = clean_vehicle_url(url)
    vehicle_id = url_cleaned.split('/')[-1]

    ## initialize browser session
    driver = firefox_driver_init(headless=headless)

    ## giant try-except because otherwise the driver will not get closed at the end
    df=None
    try:

        driver.get(url_cleaned)
        print(f"single vehicle listing [{url_cleaned}]- webpage initiated\n")

        ## capture year make model as text
        try:
            year_make_model_element = WebDriverWait(driver, wait_time).until(
                EC.visibility_of_element_located(
                    (By.CSS_SELECTOR, 'h1[data-cmp="heading"]#vehicle-details-heading')
                )
            )
            year_make_model = year_make_model_element.text
            print(f'year_make_model - success')
        except (NoSuchElementException, TimeoutException):
            year_make_model = None
            print(f'year_make_model - fail')
            


        ## listing price
        # todo if this fails can try to get it from the image gallery
        '''
        <div class="modal-header"><h4 id="modal-title" class="modal-title">Used 2022 Porsche Taycan<span class="text-normal"> - $60,944</span>
        </h4><button data-cmp="closeButton" class="close" aria-label="Close" type="button"><span class="glyphicon glyphicon-remove"></span></button></div>
        '''
        try:
            list_price_element = driver.find_element(By.CSS_SELECTOR, '[data-cmp="listingPrice"]')
            list_price = clean_text_remove_newline(list_price_element.text)
            list_price = list_price.split('^')[-1].strip()
            print(f'list_price - success')
        except (NoSuchElementException, TimeoutException):
            list_price = None
            print(f'list_price - fail')


        ## VIN
        try:
            vin_element = driver.find_element(By.CSS_SELECTOR, 'span.display-block.display-sm-inline-block')
            vin = get_clean_vin(vin_element.text)
            print(f'vin - success')
        except (NoSuchElementException, TimeoutException):
            vin = None
            print(f'vin - fail')


        ## listing details e.g. mileage
        try:
            listing_detail_element = driver.find_element(By.CSS_SELECTOR, 'ul[data-cmp="listColumns"].list.columns.columns-1')
            listing_detail = listing_detail_element.text
            listing_detail = clean_text_remove_newline(listing_detail)
            print(f'listing_detail - success')
        except (NoSuchElementException, TimeoutException):
            listing_detail = None
            print(f'listing_detail - fail')

        try:
            listing_narrative_element = driver.find_element(By.CSS_SELECTOR, '[data-cmp="seeMore"]')
            listing_narrative = clean_text_remove_newline(listing_narrative_element.text)
            print(f'listing_narrative - success')
        except (NoSuchElementException, TimeoutException):
            listing_narrative = None
            print(f'listing_narrative - fail')


        ## also attempt to just get the header image
        try:
            header_image_url = driver.find_element(By.CSS_SELECTOR, '[data-cmp="responsiveImage"]').get_attribute('src')
            print(f'header image - success')
        except (NoSuchElementException, TimeoutException):
            header_image_url = None
            print(f'header image - fail')

        ## attept to use View ALl Media button if it exists
        try:
            image_urls = get_image_urls_from_view_all_media_button(driver)
            print(f'get_image_urls_from_view_all_media_button - success - {len(image_urls)}')

        except (NoSuchElementException, TimeoutException):
            image_urls=[]
            print(f'get_image_urls_from_view_all_media_button - fail')

        if header_image_url is not None:
            image_urls+=[str(header_image_url)]

        message = f'found {len(image_urls)} images'
        print(message)

        if quit:
            driver.quit()
            driver = None

        ## save to disk
        df = pd.DataFrame({'vehicle_image_url': image_urls})
        df['vehicle_id']= vehicle_id
        df['url']= url_cleaned
        df['vin']= vin
        df['year_make_model'] = year_make_model
        df['list_price'] = list_price
        df['listing_details'] = listing_detail
        df['listing_narrative'] = listing_narrative
        df=df.drop_duplicates()
        print(df.head(1).T)
        filepath = f'{parent_directory_url_csvs}{vehicle_id}.csv'
        message=f'saving to {filepath}'
        print(message)
        df.to_csv(filepath, index=False, quoting=csv.QUOTE_ALL)

    except Exception as e:
        error_message = traceback.format_exc()
        print(f"find_vehicle_image_urls [{url}] Error Traceback:\n", error_message)
        driver.quit()
        driver = None


    return df, driver



def find_listings_for_make_model(vehicle_info:dict, driver=None, quit:bool=True) -> pd.DataFrame:

    make, model = vehicle_info['make'], vehicle_info['model']
    all_zip_codes = load_zipcodes()
    zipcode = vehicle_info.get('zipcode')
    if zipcode is None:
        zipcode = random.choice(all_zip_codes)



    ## user args specify make, model, and search zipcode
    # the url specifies a nationwide search, sorted by distance
    url_template = f'https://www.autotrader.com/cars-for-sale/{make}/{model}/{zipcode}?searchRadius=0&sortBy=distanceASC'
    url = url_template.format(make=make, model=model,zipcode=zipcode)
    print(f"attempt to access webpage [{url}]")

    ## initialize browser session - unless of course it is already initialized
    if driver is None:

        driver = firefox_driver_init(headless=headless)
        driver.get(url)
        print(f"make/model search [{url}] webpage initiated\n")

    ## think about editing the search distance (default is only 50 which may be inadequate for rare cars)
    # instead of interacting w the dropdown menu like a caveperson, I can just manipulate the url (big brain)
    # nationwide= True
    # if nationwide:
    #     dropdown = Select(driver.find_element(By.NAME, "searchRadius"))
    #     dropdown.select_by_value("0") # corresponds to nationwide. otherwise choose a # of miles


    # print("make/model search - initiate scrolling")
    scroll_down_incrementally(driver)
    vehicle_listing_links = find_vehicle_listing_links(driver)

    ##  check if I got all the listings or not
    result_count_expected = get_search_result_count(driver)
    result_count_actual = len(vehicle_listing_links)
    _diff = result_count_expected - result_count_actual
    if _diff >0:
        message = f'WARNING found {result_count_actual} / {result_count_expected} listings ; {_diff} are missing '
    else:
        # message = f'Found all {result_count_expected} listings'
        message = f'found {result_count_actual} / {result_count_expected} listings'
    print(message)

    current_url = driver.current_url

    ## todo think about a way to navigate to page 2, page3 of search results
    # this can be done by clicking or it can be done by pre-populating the url in a certain way e.g. ?firstRecord=50
    # https://www.autotrader.com/cars-for-sale/ford/taurus/san-diego-ca?firstRecord=50&searchRadius=0&sortBy=distanceASC&zip=92101

    ## prepare result DF
    if quit:
        driver.quit()
        driver = None

    df = pd.DataFrame(vehicle_listing_links, columns=['listing_header', 'url'])
    df['search_url'] = current_url
    ts = int(time.time())
    df['make']= make
    df['model']= model
    df['search_timestamp'] =ts
    df['search_metadata'] = str(vehicle_info)
    print(df.head(1).T)
    filename =f'search_results_{make}_{model}_{ts}'
    filepath = f'{parent_directory_url_csvs}{filename}.csv'
    message=f'saving to {filepath}'
    print(message)
    df.to_csv(filepath, index=False, quoting=csv.QUOTE_ALL)

    return df, driver


def compile_search_results_df() ->pd.DataFrame:
    '''
    :return:
    DF where each row represents on vehicle listing with a unique url and vehicle id

    Also contains make/model of the SEARCH from which the url was captured
    NB: This might not be the make/model of the actual vehicle;
     The vehicle listing page is the ultimate source of truth on the make/model of the vehicle

    '''
    ## compile search results from multiple csvs intoo one df
    # folder="/Users/levgolod/Projects/car_classifier/data/autotrader/vehicle_metadata/"
    folder=parent_directory_url_csvs
    pattern=r'search_results_.*[0-9]*csv'
    usecols=['listing_header',
     'url',
     'search_url',
     'make',
     'model',
     'search_timestamp',
     'search_metadata']
    # bigdf= pd.DataFrame(columns=usecols)
    bigdf= None

    files = [x for x in os.listdir(folder) if bool(re.search(pattern, x))]

    # file=files[0]
    for file in files:
        print(file)
        try:
            df=pd.read_csv(folder+file, usecols=usecols)
            df=df[usecols]
            df['filename']=file
        except ValueError:
            print(f'File {file} wrong columns, skipping')
            continue


        bigdf = pd.concat([bigdf, df], ignore_index=True) if bigdf is not None else df
        # bigdf=bigdf.drop_duplicates(subset=['url'])
        print(bigdf.shape)

    bigdf['url_clean'] = bigdf['url'].apply(clean_vehicle_url)
    bigdf['vehicle_id'] = bigdf['url_clean'].apply(lambda x: str(x).split('/')[-1] )

    bigdf.head()
    return bigdf

def compile_image_urls_df():
    folder=parent_directory_url_csvs
    pattern=r'^[0-9]*\.csv'
    dtypes={'vehicle_image_url': 'str',
    'vehicle_id': 'int',
    'url': 'str',
    'vin': 'str',
    'year_make_model': 'str',
    'list_price': 'str',
    'listing_details': 'str',
    'listing_narrative': 'str'}
    bigdf=None
    usecols = list(dtypes.keys())
    1

    files = [x for x in os.listdir(folder) if bool(re.search(pattern, x))]

    for file in files:
        print(file)
        try:
            df=pd.read_csv(folder+file, usecols=usecols, dtype=dtypes)
            df=df[usecols]
            df['filename']=file
        except ValueError:
            print(f'File {file} wrong columns, skipping')
            continue


        bigdf = pd.concat([bigdf, df], ignore_index=True) if bigdf is not None else df
        print(bigdf.shape)

    return bigdf


if __name__ == '__main__':


    # try out process_vehicle_webpage
    urls = [
        # 'https://www.autotrader.com/cars-for-sale/vehicle/725617155',
        'https://www.autotrader.com/cars-for-sale/vehicle/739886024',
        # 'https://www.autotrader.com/cars-for-sale/vehicle/739471281?LNX=SPGOOGLEBRANDPLUSMAKE&city=San%20Diego&ds_rl=1289689&gad_source=1&gclid=CjwKCAiA5Ka9BhB5EiwA1ZVtvHjCtPelBybjmSVqEfXhXQ4FoLUB_-DHe9sxqf7bMrntLKD3J1AJKRoCz-8QAvD_BwE&gclsrc=aw.ds&listingType=USED&makeCode=TOYOTA&modelCode=CAMRY&referrer=%2Ftoyota%2Fcamry%3FLNX%3DSPGOOGLEBRANDPLUSMAKE%26ds_rl%3D1289689%26gad_source%3D1%26gclid%3DCjwKCAiA5Ka9BhB5EiwA1ZVtvHjCtPelBybjmSVqEfXhXQ4FoLUB_-DHe9sxqf7bMrntLKD3J1AJKRoCz-8QAvD_BwE%26gclsrc%3Daw.ds%26utm_campaign%3Dat_na_na_national_evergreen_roi_na_na%26utm_content%3Dkeyword_text_na_na_na_spgooglebrandplusmake_na%26utm_medium%3Dsem_brand-plus_perf%26utm_source%3DGOOGLE%26utm_term%3Dautotrader%2520toyota%2520camry&state=CA&utm_campaign=at_na_na_national_evergreen_roi_na_na&utm_content=keyword_text_na_na_na_spgooglebrandplusmake_na&utm_medium=sem_brand-plus_perf&utm_source=GOOGLE&utm_term=autotrader%20toyota%20camry&zip=92101&clickType=listing',
        # 'https://www.autotrader.com/cars-for-sale/vehicle/736665996?city=Irvine&listingType=USED&makeCode=POR&modelCode=PORTAYCAN&referrer=%2Fporsche%2Ftaycan%3F&state=CA&zip=92604&clickType=listing'
    ]

    for url in urls:
        print(url)
        with timethis():
            process_vehicle_webpage(url, False)


    # ## try out find_listings_for_make_model
    # vehicles_to_search = [
    #     # {'make':'alfa-romeo','model':'4c'},
    #     {'make':'porsche','model':'taycan'},
    #     # {'make': 'hyundai','model':'ioniq5'}
    #     # {'make': 'hyundai','model':'asgadgadhadhsfqtdhg'}
    # ]
    # for vehicle_info in vehicles_to_search:
    #     print(vehicle_info)
    #     with timethis():
    #         find_listings_for_make_model(vehicle_info, False)


    '''
    https://www.autotrader.com/cars-for-sale/ford/taurus/san-diego-ca?firstRecord=50&searchRadius=0&sortBy=distanceASC&zip=92101
    '''