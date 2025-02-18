import selenium
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException

from webdriver_manager.chrome import ChromeDriverManager
from random_user_agent.user_agent import UserAgent
from random_user_agent.params import SoftwareName, OperatingSystem
import random
import datetime as dt
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
chromedriver_path = "/usr/local/bin/chromedriver"
chrome_path = "/Applications/chrome-mac-x64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
browser='chrome'
parent_directory_url_csvs='/Users/levgolod/Projects/car_classifier/data/autotrader/vehicle_metadata/'
parent_directory_images='/Users/levgolod/Projects/car_classifier/data/autotrader/vehicle_images/'
vehicle_url_template='https://www.autotrader.com/cars-for-sale/vehicle/{vehicle_id}'
wait_time=10
scroll_pause_time = 0.2  # Adjust based on load time
scroll_distance = 500 # pixels
max_scrolls = 100
# headless=True
headless=False
####################################


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



def chrome_driver_init():
    import webdriver_manager
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    chrome_path = "/Applications/chrome-mac-x64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
    options = webdriver.ChromeOptions()
    options.binary_location = chrome_path
    # options.page_load_strategy = "none" # too aggressive, not worth it
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get("https://www.google.com")
    return driver



def get_vehicle_make_model_list():
    vehicles = [
        {'make': 'bmw', 'model': '4-series', 'body_style': 'sedan'},
        {'make': 'bmw', 'model': '3-series', 'body_style': 'sedan'},
        {'make': 'bmw', 'model': 'x5', 'body_style': 'suv'},
        {'make': 'bmw', 'model': 'z4', 'body_style': 'sportscar'},
        {'make': 'ford', 'model': 'taurus', 'body_style': 'sedan'},
        {'make': 'ford', 'model': 'f150', 'body_style': 'truck'},
        {'make': 'ford', 'model': 'explorer', 'body_style': 'suv'},
        {'make': 'honda', 'model': 'ridgeline', 'body_style': 'truck'},
        {'make': 'honda', 'model': 'odyssey', 'body_style': 'van'},
        {'make': 'honda', 'model': 'accord', 'body_style': 'sedan'},
        {'make': 'honda', 'model': 'civic', 'body_style': 'sedan'},
        {'make': 'honda', 'model': 'passport', 'body_style': 'suv'},
        {'make': 'toyota', 'model': 'camry', 'body_style': 'sedan'},
        {'make': 'toyota', 'model': 'crown', 'body_style': 'sedan'},
        {'make': 'toyota', 'model': 'tacoma', 'body_style': 'truck'},
        {'make': 'toyota', 'model': 'tundra', 'body_style': 'truck'},
        {'make': 'toyota', 'model': 'corolla', 'body_style': 'sedan'},
        # {'make': 'volvo', 'model': 'v90', 'body_style': 'wagon'},
        # {'make': 'volvo', 'model': 'c30', 'body_style': 'wagon'},
    ]
    return vehicles


def load_geog_df() -> pd.DataFrame:
    '''
    Source https://simplemaps.com/data/us-zips
    :return:
    beverly-hills-ca
    '''
    # zips_filepath='~/Projects/car_classifier/data/simplemaps_uszips_basicv1.90/uszips.csv'
    zips_filepath='./data/simplemaps_uszips_basicv1.90/uszips.csv'
    zips_df = pd.read_csv(zips_filepath)
    zips_df['zip'] =zips_df['zip'].astype(str).str.zfill(5)
    zips_df['city_state_lower'] = zips_df['city'].str.lower().str.replace(' ','-') + '-' + \
                                  zips_df['state_id'].str.lower()
    return zips_df


def load_zipcodes() -> list:
    '''
    Source https://simplemaps.com/data/us-zips
    :return:
    '''
    zips_df = load_geog_df()
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
    try:
        result_count_element = driver.find_element(By.CSS_SELECTOR, '[data-cmp="resultsCount"]')
        result_count_text = result_count_element.text
        result_count = int(result_count_text.split(' ')[0].replace(',',''))
        return result_count
    except:
        return -1


def find_image_urls(driver) -> list:
    # get the image urls. There are multiple sizes available.
    # I used to 'keep it one hundred', but due to inflation I am gonna 'keep it 500'
    image_url_prefix = 'https://images.autotrader.com/scaler/500'
    return list(filter(
        lambda x: str(x).startswith(image_url_prefix),
        str(driver.page_source).split()
    ))

def find_image_urls_v2(driver) -> list:
    # more flexible
    image_url_prefix = 'https://images.autotrader.com/'
    urls = list(filter(
        lambda x: str(x).startswith(image_url_prefix),
        str(driver.page_source).split()
    ))
    df_urls = pd.DataFrame({'url': urls})
    df_urls['filename'] = df_urls['url'].apply(
        lambda x: os.path.basename(x).split('.')[0]
    )
    # for each unique filename, keep the url w the largest image size e.g. 500
    df_urls = df_urls.sort_values(by=['filename', 'url'], ascending=[1, 0])
    df_urls.reset_index(drop=True, inplace=True)
    df_urls2 = df_urls.groupby('filename').head(1)
    # print(len(df_urls),len(df_urls2))
    df_urls2
    return list(df_urls2['url'])



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


def wait_for_scrollbar(driver, timeout=wait_time):
    """
    Waits for the scrollbar to appear on the webpage.

    :param driver: Selenium WebDriver instance
    :param timeout: Maximum time (in seconds) to wait for the scrollbar
    :return: True if scrollbar exists, False otherwise
    """
    try:
        WebDriverWait(driver, timeout).until(
            lambda d: driver.execute_script("return document.body.scrollHeight") > driver.execute_script("return window.innerHeight")
        )
        print("Scrollbar detected!")
        return True
    except:
        print("No scrollbar found after waiting.")
        return False


def get_scroll_percentage(driver):
    """Returns how far down the page is scrolled as a percentage."""
    scroll_top = driver.execute_script("return window.scrollY;")  # Current scroll position
    scroll_height = driver.execute_script("return document.documentElement.scrollHeight;")  # Total scroll height
    client_height = driver.execute_script("return window.innerHeight;")  # Viewport height

    # Calculate percentage (0 to 100)
    scroll_percentage = (scroll_top / (scroll_height - client_height)) * 100
    return round(scroll_percentage, 2)


def scroll_down_incrementally(driver, scroll_distance=scroll_distance, wait=0.1, verbose=False):
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


def check_for_site_unavailable(driver):
    return 'site is currently unavailable' in str(driver.page_source)



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
    image_urls = find_image_urls_v2(driver)
    last_length = len(image_urls)
    i = 0
    n_useless_scrolls=0
    while True:
        # Scroll down by a small amount
        driver.execute_script(f"arguments[0].scrollTop += {scroll_distance};", scroll_panel)
        # print(f'they see me scrollin, they hatin ({i})')
        i += 1
        time.sleep(scroll_pause_time)  # Wait for content to load
        image_urls = find_image_urls_v2(driver)
        new_length = len(image_urls)
        n_useless_scrolls += int(new_length == last_length)

        if (n_useless_scrolls >= 5):
            print(f'the futility is unbearable {i} {n_useless_scrolls}')
            break

        if (i >= max_scrolls):
            print(f'cant scroll anymore boss {i} {n_useless_scrolls}')
            break

        last_length = new_length

    return image_urls



def driver_init(browser:str=browser):
    if browser == 'chrome':
        return chrome_driver_init()
    elif browser == 'firefox':
        return firefox_driver_init(headless=headless)

    return None


def process_vehicle_webpage(url:str, quit:bool=True) -> tuple:

    url_cleaned = clean_vehicle_url(url)
    vehicle_id = url_cleaned.split('/')[-1]

    ## initialize browser session
    # driver = firefox_driver_init(headless=headless)
    driver = driver_init()

    ## giant try-except because otherwise the driver will not get closed at the end
    df=pd.DataFrame()
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

        page_unavailable = check_for_site_unavailable(driver)

        if  page_unavailable:
            message=f' page_unavailable {page_unavailable}; that is not good!'
            print(message)
            driver.quit()
            return pd.DataFrame(), None


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
        if len(image_urls) >0:
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
        else:
            print(f'no images; not saving csv')


    except Exception as e:
        error_message = traceback.format_exc()
        print(f"process_vehicle_webpage [{url}] Error Traceback:\n", error_message)
        driver.quit()
        driver = None
        df = pd.DataFrame()


    return df, driver


def capture_listings_from_current_page(driver) -> tuple:
    '''
    :param driver:
    :return: (pd.DataFrame, driver)
    '''

    # time.sleep(wait_time) # be smarter and wait until things are actually loaded.
    scrollbar_exists = wait_for_scrollbar(driver)
    page_unavailable = check_for_site_unavailable(driver)

    if (not scrollbar_exists) or page_unavailable:
        message = f'scrollbar failed to load {not scrollbar_exists} and/or page_unavailable {page_unavailable}; that is not good!'
        print(message)
        driver.quit()
        return pd.DataFrame(), None

    scroll_down_incrementally(driver)
    vehicle_listing_links = find_vehicle_listing_links(driver)

    ##  check if I got all the listings or not
    result_count_expected = get_search_result_count(driver)
    result_count_actual = len(vehicle_listing_links)
    _diff = result_count_expected - result_count_actual
    if _diff > 0:
        message = f'WARNING found {result_count_actual} / {result_count_expected} listings ; {_diff} are missing '
    else:
        message = f'found {result_count_actual} / {result_count_expected} listings'
    print(message)

    current_url = driver.current_url

    df = pd.DataFrame(vehicle_listing_links, columns=['listing_header', 'url'])
    df = df.drop_duplicates()
    df['search_url'] = current_url
    search_timestamp = int(time.time())
    df['search_timestamp'] = search_timestamp
    print(df.head(1).T)
    return df, driver


def find_listings_for_make_model(vehicle_info:dict, driver=None, quit:bool=True) -> tuple:

    df_geog = load_geog_df()
    search_radius=0 # 0 corresponds to nationwide
    sort_by='distanceASC' # e.g. 'datelistedDESC','distanceASC'

    make, model = vehicle_info['make'], vehicle_info['model']
    first_record = int(vehicle_info.get('first_record', 0)) # 5000
    zipcode = vehicle_info.get('zipcode')
    city_state = vehicle_info.get('city_state_lower')
    if zipcode is None:
        print('choose random location within USA')
        location = list(df_geog.sample(n=1).to_dict('records'))[0]
        zipcode = location['zip']
        city_state = location['city_state_lower']


    ## you can do a lot by messing with the url, you know. really quite a bit. not everyone knows that.
    url_template = 'https://www.autotrader.com/cars-for-sale/' + \
        f'{make}/{model}/{city_state}?firstRecord={first_record}&searchRadius={search_radius}&sortBy={sort_by}&zip={zipcode}'
    # https://www.autotrader.com/cars-for-sale/bmw/3-series/houston-tx?firstRecord=5000&searchRadius=100&sortBy=distanceASC&zip=77038

    url = url_template.format(make=make, model=model,zipcode=zipcode)
    print(f"attempt to access webpage [{url}]")
    #

    ## initialize browser session - unless of course it is already initialized
    if driver is None:
        # driver = firefox_driver_init(headless=headless)
        # driver = chrome_driver_init()
        driver = driver_init()

    ## giant try-except because otherwise the driver will not get closed at the end
    df=pd.DataFrame()
    try:

        driver.get(url)
        print(f"make/model search [{url}] webpage initiated\n")

        df,driver = capture_listings_from_current_page(driver)
        assert len(df)>0

        df['make'] = make
        df['model'] = model
        df['search_metadata'] = str(vehicle_info)
        search_timestamp = list(df['search_timestamp'])[0]

        ## prepare result DF
        if quit:
            driver.quit()
            driver = None

        filename =f'search_results_{make}_{model}_{search_timestamp}'
        filepath = f'{parent_directory_url_csvs}{filename}.csv'
        message=f'saving to {filepath}'
        print(message)
        df.to_csv(filepath, index=False, quoting=csv.QUOTE_ALL)

    except Exception as e:
        error_message = traceback.format_exc()
        print(f"find_listings_for_make_model [{url}] Error Traceback:\n", error_message)
        driver.quit()
        driver = None
        df = pd.DataFrame()

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
    for file in files:
        # print(file)
        try:
            df=pd.read_csv(folder+file, usecols=usecols)
            df=df[usecols]
            df['filename']=file
        except ValueError:
            print(f'File {file} wrong columns, skipping')
            continue


        bigdf = pd.concat([bigdf, df], ignore_index=True) if bigdf is not None else df
        # bigdf=bigdf.drop_duplicates(subset=['url'])
        # print(bigdf.shape)

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
        # print(file)
        try:
            df=pd.read_csv(folder+file, usecols=usecols, dtype=dtypes)
            df=df[usecols]
            df['filename']=file
        except ValueError:
            print(f'File {file} wrong columns, skipping')
            continue


        bigdf = pd.concat([bigdf, df], ignore_index=True) if bigdf is not None else df
        # print(bigdf.shape)

    return bigdf


if __name__ == '__main__':


    # try out process_vehicle_webpage
    urls = [
        # 'https://www.autotrader.com/cars-for-sale/vehicle/725617155',
        # 'https://www.autotrader.com/cars-for-sale/vehicle/739886024',
        # 'https://www.autotrader.com/cars-for-sale/vehicle/739471281?LNX=SPGOOGLEBRANDPLUSMAKE&city=San%20Diego&ds_rl=1289689&gad_source=1&gclid=CjwKCAiA5Ka9BhB5EiwA1ZVtvHjCtPelBybjmSVqEfXhXQ4FoLUB_-DHe9sxqf7bMrntLKD3J1AJKRoCz-8QAvD_BwE&gclsrc=aw.ds&listingType=USED&makeCode=TOYOTA&modelCode=CAMRY&referrer=%2Ftoyota%2Fcamry%3FLNX%3DSPGOOGLEBRANDPLUSMAKE%26ds_rl%3D1289689%26gad_source%3D1%26gclid%3DCjwKCAiA5Ka9BhB5EiwA1ZVtvHjCtPelBybjmSVqEfXhXQ4FoLUB_-DHe9sxqf7bMrntLKD3J1AJKRoCz-8QAvD_BwE%26gclsrc%3Daw.ds%26utm_campaign%3Dat_na_na_national_evergreen_roi_na_na%26utm_content%3Dkeyword_text_na_na_na_spgooglebrandplusmake_na%26utm_medium%3Dsem_brand-plus_perf%26utm_source%3DGOOGLE%26utm_term%3Dautotrader%2520toyota%2520camry&state=CA&utm_campaign=at_na_na_national_evergreen_roi_na_na&utm_content=keyword_text_na_na_na_spgooglebrandplusmake_na&utm_medium=sem_brand-plus_perf&utm_source=GOOGLE&utm_term=autotrader%20toyota%20camry&zip=92101&clickType=listing',
        # 'https://www.autotrader.com/cars-for-sale/vehicle/736665996?city=Irvine&listingType=USED&makeCode=POR&modelCode=PORTAYCAN&referrer=%2Fporsche%2Ftaycan%3F&state=CA&zip=92604&clickType=listing'
    ]

    for url in urls:
        print(url)
        with timethis():
            process_vehicle_webpage(url, False)


    # ## try out find_listings_for_make_model
    vehicles_to_search = [
        # {'make':'alfa-romeo','model':'4c'},
        # {'make':'porsche','model':'taycan'},
        # {'make': 'ford', 'model': 'f150','zipcode':'92604','city_state_lower':'irvine-ca'},
        # {'make': 'ford', 'model': 'f150','zipcode':'83223','city_state_lower':'bloomington-id'},
        # {'make': 'bmw', 'model': '4-series','zipcode':'92604','city_state_lower':'irvine-ca'},
        # {'make': 'ford', 'model': 'taurus','zipcode':'92101','city_state_lower':'san-diego-ca'},
        {'make': 'ford', 'model': 'f150','zipcode':'92101','city_state_lower':'san-diego-ca','first_record':100},
        # {'make': 'hyundai','model':'ioniq5'}
        # {'make': 'hyundai','model':'asgadgadhadhsfqtdhg'}
    ]
    for vehicle_info in vehicles_to_search:
        print(vehicle_info)
        with timethis():
            df,driver = find_listings_for_make_model(vehicle_info,quit=False)
        time.sleep(60)


    '''
    https://www.autotrader.com/cars-for-sale/ford/taurus/san-diego-ca?firstRecord=50&searchRadius=0&sortBy=distanceASC&zip=92101
    https://www.autotrader.com/cars-for-sale/bmw/3-series/san-diego-ca?firstRecord=50&searchRadius=100&sortBy=distanceASC&zip=92101
    https://www.autotrader.com/cars-for-sale/bmw/3-series/san-diego-ca?firstRecord=5000&searchRadius=100&sortBy=distanceASC&zip=92101
    https://www.autotrader.com/cars-for-sale/bmw/3-series/houston-tx?firstRecord=5000&searchRadius=100&sortBy=distanceASC&zip=77038
    https://www.autotrader.com/cars-for-sale/ford/taurus/san-diego-ca?firstRecord=50&searchRadius=0&sortBy=distanceASC&zip=92101
    https://www.autotrader.com/cars-for-sale/ford/taurus/san-diego-ca?firstRecord=5000&searchRadius=0&sortBy=datelistedDESC&zip=92101
    
    IP 98.176.105.96
    
    ## todo think about a way to navigate to page 2, page3 of search results
    # this can be done by clicking or it can be done by pre-populating the url in a certain way e.g. ?firstRecord=50
    # https://www.autotrader.com/cars-for-sale/ford/taurus/san-diego-ca?firstRecord=50&searchRadius=0&sortBy=distanceASC&zip=92101
    '''



# def chrome_driver_init():
#     random_user_agent = create_random_user_agent()
#     options = webdriver.ChromeOptions()
#     options.binary_location = chrome_path
#     options.add_argument(f"user-agent={random_user_agent}")
#     driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
#     driver.get("https://www.google.com")
#     return driver