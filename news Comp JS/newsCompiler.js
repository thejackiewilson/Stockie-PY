const axios = require('axios');
const cheerio = require('cheerio');
const express = require('express');
const puppeteer = require('puppeteer');
const app = express();
app.set('view engine', 'ejs');
app.set('views', './views');


// URLs of the financial news sources
const sources = [
  {
    name: 'Federal Reserve',
    url: 'https://www.federalreserve.gov/newsevents/pressreleases.htm',
    parse: async () => {
      const browser = await puppeteer.launch();
      const page = await browser.newPage();
      await page.goto('https://www.federalreserve.gov/newsevents/pressreleases.htm');

      const newsItems = await page.evaluate(() => {
        const items = [];
        const elements = document.querySelectorAll('.panel-body ul.list-unstyled.panel-body__list a');
        for (const el of elements) {
          items.push({
            title: el.innerText,
            url: `https://www.federalreserve.gov${el.getAttribute('href')}`,
          });
        }
        return items;
      });

      await browser.close();
      return newsItems;
    },
  },
];

// Fetch and parse news from sources
async function fetchNews() {
  const allNews = [];

  for (const source of sources) {
    try {
      const newsItems = await source.parse();
      allNews.push(...newsItems);
    } catch (error) {
      console.error(`Failed to fetch news from ${source.name}:`, error.message);
    }
  }

  return allNews;
}

// Define the route for the homepage
app.get('/', async (req, res) => {
    try {
      const news = await fetchNews();
      res.render('index', { news });
    } catch (error) {
      console.error('Failed to fetch news:', error.message);
      res.status(500).send('Failed to fetch news');
    }
  });  

// Start the server
const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`Server is running at http://localhost:${port}`);
});
