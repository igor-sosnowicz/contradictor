"""
Script for downloading and building a benchmark dataset of raw HTML pages
grouped by interpretative frames.
"""

import time
import urllib.request
from pathlib import Path

from loguru import logger

DATA_DIR = Path(__file__).resolve().parent / "fixtures"

URLS_BY_FRAME = {
    "Economic": [
        "https://www.ft.com/global-economy",
        "https://www.wsj.com/economy",
        "https://www.bloomberg.com/economics",
        "https://www.economist.com/finance-and-economics",
        "https://www.cnbc.com/economy/",
        "https://www.reuters.com/markets/wealth/",
        "https://www.worldbank.org/en/news",
    ],
    "Capacity and Resources": [
        "https://www.iea.org/news",
        "https://www.bp.com/en/global/corporate/energy-economics.html",
        "https://www.fao.org/newsroom/en",
        "https://www.usgs.gov/news",
        "https://www.wateraid.org/uk/news-and-stories",
        "https://www.mining.com/news/",
        "https://www.eia.gov/todayinenergy/",
    ],
    "Morality": [
        "https://www.vaticannews.va/en.html",
        "https://www.theosthinktank.co.uk/comment",
        "https://www.ethicsandculture.com/",
        "https://www.philosophyforlife.org/blog",
        "https://www.humanists.uk/news/",
        "https://www.dalailama.com/news",
        "https://www.christianitytoday.com/news/",
    ],
    "Fairness and Equality": [
        "https://www.unwomen.org/en/news-stories",
        "https://www.amnesty.org/en/latest/news/",
        "https://www.hrw.org/news",
        "https://www.oxfam.org/en/press-releases",
        "https://www.aclu.org/news",
        "https://www.equalitynow.org/news_and_insights/",
        "https://eurekalert.org",
    ],
    "Constitutionality and Legality": [
        "https://www.scotusblog.com/",
        "https://www.law.com/nationallawjournal/",
        "https://www.jurist.org/news/",
        "https://www.supremecourt.gov/",
        "https://www.harvardlawreview.org/",
        "https://www.oyez.org/",
        "https://www.icj-cij.org/en/news",
    ],
    "Policy Prescription and Evaluation": [
        "https://www.brookings.edu/blog/policy2020/",
        "https://www.rand.org/blog.html",
        "https://www.cato.org/blog",
        "https://www.heritage.org/commentary",
        "https://www.cfr.org/blog",
        "https://www.chathamhouse.org/publications/expert-comment",
        "https://www.ceps.eu/publications/",
    ],
    "Crime and Justice": [
        "https://www.themarshallproject.org/",
        "https://www.fbi.gov/news",
        "https://www.interpol.int/en/News-and-Events",
        "https://www.justice.gov/news",
        "https://www.prisonpolicy.org/news/",
        "https://www.sentencingproject.org/news/",
        "https://www.crimestoppers-uk.org/news-campaigns",
    ],
    "Security and Defense": [
        "https://www.defensenews.com/",
        "https://www.janes.com/news",
        "https://www.nato.int/cps/en/natohq/news.htm",
        "https://www.defense.gov/News/",
        "https://www.military.com/daily-news",
        "https://www.iiss.org/online-analysis",
        "https://www.army.mil/news",
    ],
    "Health and Safety": [
        "https://www.who.int/news-room",
        "https://www.cdc.gov/media/index.html",
        "https://www.thelancet.com/journals/lancet/onlinefirst",
        "https://www.nejm.org/medical-articles/news",
        "https://www.nih.gov/news-events/news-releases",
        "https://www.fda.gov/news-events/press-announcements",
        "https://bmj.com",
    ],
    "Quality of Life": [
        "https://www.livehappy.com/",
        "https://www.mindbodygreen.com/",
        "https://www.wellandgood.com/",
        "https://www.psychologytoday.com/us/blog",
        "https://www.architecturaldigest.com/architecture",
        "https://www.travelandleisure.com/",
        "https://www.goodnewsnetwork.org/",
    ],
    "Cultural Identity": [
        "https://www.nationalgeographic.com/history-culture",
        "https://www.smithsonianmag.com/history-archaeology/",
        "https://www.bbc.com/culture",
        "https://www.hyperallergic.com/",
        "https://www.theartnewspaper.com/",
        "https://www.history.com/news",
        "https://nationaltrust.org.uk",
    ],
    "Public Opinion": [
        "https://www.pewresearch.org/",
        "https://news.gallup.com/home.aspx",
        "https://www.ipsos.com/en/news-polls",
        "https://yougov.co.uk/",
        "https://www.pollingreport.com/",
        "https://www.maristpoll.marist.edu/",
        "https://www.monmouth.edu/polling-institute/",
    ],
    "Political": [
        "https://www.politico.com/",
        "https://www.thehill.com/",
        "https://www.axios.com/",
        "https://www.rollcall.com/",
        "https://www.realclearpolitics.com/",
        "https://www.politico.eu/",
        "https://spectator.co.uk",
    ],
    "External Regulation and Reputation": [
        "https://www.wto.org/english/news_e/news_e.htm",
        "https://www.un.org/press/en",
        "https://www.oecd.org/newsroom/",
        "https://www.imf.org/en/News",
        "https://www.transparency.org/en/news",
        "https://www.ftc.gov/news-events/news",
        "https://www.europol.europa.eu/newsroom",
    ],
    "Other": [
        "https://www.wired.com/category/science/",
        "https://www.techcrunch.com/",
        "https://www.space.com/news",
        "https://www.nature.com/nature/news-features",
        "https://www.bbc.com/news/science_and_environment",
    ],
}


def download_dataset() -> None:
    """
    Download a dataset of 100 HTML pages from predefined URLs.

    Iterates through a mapping of URLs, performs HTTP requests with standard
    browser headers, and saves the retrieved HTML content to the local data
    directory. Errors during download are logged and handled gracefully.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }

    flat_urls = [
        (frame, url) for frame, links in URLS_BY_FRAME.items() for url in links
    ]
    total = len(flat_urls)
    logger.info(f"Starting download of {total} English pages to: {DATA_DIR}\n")
    for i, (frame, url) in enumerate(flat_urls, 1):
        try:
            if not url.startswith(("http://", "https://")):
                raise ValueError(f"Unsupported URL scheme: {url}")
            req = urllib.request.Request(url, headers=headers)  # noqa: S310, added manual filtering
            with urllib.request.urlopen(req, timeout=12) as response:  # noqa: S310, added manual filtering
                html = response.read().decode("utf-8", errors="ignore")
                domain = url.split("//")[-1].split("/")[0].replace(".", "_")
                file_name = f"{i:03d}_{frame.replace(' ', '_')}_{domain}.html"
                file_path = DATA_DIR / file_name
                file_path.write_text(html, encoding="utf-8")
                logger.info(f"[{i}/{total}] [{frame}] Downloaded: {url} -> {file_name}")
                time.sleep(1.5)
        except urllib.error.URLError as e:
            logger.info(f"[{i}/{total}] [{frame}] ERROR downloading {url}: {e}")


if __name__ == "__main__":
    download_dataset()
