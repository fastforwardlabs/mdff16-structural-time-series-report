let puppeteer = require('puppeteer')
let fs = require('fs-extra')
let path = require('path')

let margin = '0.5in'

;(async () => {
  let browser = await puppeteer.launch()
  let page = await browser.newPage()
  await page.goto(`file:${path.join(__dirname, 'out/index.html')}`, {
    waitUntil: 'networkidle2',
  })

  await page.addStyleTag({
    content:
      'html { font-size: 12px; line-height: 18px; } body { padding-left: 0; } .content { padding: 0; width: 100%; } .table-of-contents { position: relative; background: white; width: 100%; height: auto; page-break-before: always; font-size: 12px !important; line-heignt: 18px !important; } .table-of-contents a { text-decoration: underline; } #toc-header { display: none !important;  } figcaption { font-size: 10px; line-height: 18px; } #report-splash { display: none; } #explanation { display: none; } table { font-size: 8px; line-height: 12px; } #report-iso { display: block; margin-top: 160px; } #html-logo { display: none !important; } #pdf-logo { display: block }',
  })
  await page.pdf({
    path: 'out/FF15-Meta-Learning-Cloudera_Fast_Forward.pdf',
    height: '8.5in',
    width: '5.5in',
    displayHeaderFooter: false,
    headerTemplate: '',
    printBackground: true,
    margin: {
      top: margin,
      left: margin,
      right: margin,
      bottom: margin,
    },
  })
  await browser.close()
})()
