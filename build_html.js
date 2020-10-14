let fs = require('fs-extra')
let path = require('path')
let md = require('markdown-it')({ typographer: true, html: true })
var implicitFigures = require('markdown-it-implicit-figures')

let deploy_location = process.argv[2]

let line = 28
let lq = line / 4
let bf = 5 / 8
let hf = 6 / 8

let rfs = bf * line
let lh = 1 / bf
let rlh = line

md.use(require('markdown-it-anchor'))
md.use(require('markdown-it-table-of-contents'), {
  includeLevel: [2, 3],
  containerHeaderHtml: `<div id="toc-header" style="display: flex; font-weight: bold; text-transform: uppercase;">
     <div><button id="toggle_contents" style="padding-left: 0.5ch; padding-right: 0.5ch; cursor: pointer; position: relative; top: -1px;">☰</button><span id="contents-label" style="margin-left: 0;"> Contents</span></div>
  </div>`,
})
md.use(require('markdown-it-container'), 'info', {})
md.use(require('markdown-it-footnote'))
md.use(implicitFigures, {
  dataType: false, // <figure data-type="image">, default: false
  figcaption: true, // <figcaption>alternative text</figcaption>, default: false
  tabindex: false, // <figure tabindex="1+n">..., default: false
  link: false, // <a href="img.png"><img src="img.png"></a>, default: false
})

let hcounter = `
h1, h2, h3, h4, h5, h6, button { font-size: inherit; line-height: inherit; font-style: inherit; font-weight: inherit; margin: 0; font-feature-settings: "tnum"; border: none; background: transparent; padding: 0;  }
button:focus, button:hover {
  background: rgba(0,0,0,0.125);
  outline: none;
}
h1 {
  font-size: ${line * 2 * hf}px;
  line-height: ${line * 2}px;
  font-weight: bold;
  margin-top: ${lq * 2}px;
  margin-bottom: ${lq * 2}px;
}
h2 {
  font-size: ${line * 1.5 * hf}px;
  line-height: ${line * 1.5}px;
  font-weight: bold;
  margin-top: ${rlh * 1}px;
  margin-bottom: ${lq * 2}px;
}
h3 {
  font-size: ${line * 1.25 * hf}px;
  line-height: ${line * 1.25}px;
  font-weight: bold;
  margin-top: ${lq * 2}px;
  margin-bottom: ${lq * 2}px;
}
h4 {
  font-size: ${line * 1 * hf}px;
  line-height: ${line * 1}px;
  font-weight: bold;
  margin-top: ${lq * 2}px;
  margin-bottom: ${lq * 2}px;
}
h5 {
  font-size: ${line * 1 * bf}px;
  line-height: ${line * 1}px;
  margin-top: ${lq * 2}px;
  margin-bottom: ${lq * 2}px;
  font-weight: bold;
}
h6 {
  font-size: ${line * 1 * bf}px;
  line-height: ${line * 1}px;
  margin-top: ${lq * 2}px;
  margin-bottom: ${lq * 2}px;
  font-style: italic;
}
p {
  margin-bottom: ${lq * 2}px;
}
.content {
  position: relative;
  }
figure {
  margin: 0;
  margin-top: ${lq * 2}px;
  margin-bottom: ${lq * 4}px;
  display: block;
  position: relative;
  page-break-inside: avoid;
}
blockquote {
  margin: 0;
   margin-top: ${lq * 2}px;
  margin-bottom: ${lq * 2}px;
margin-left: 2ch;
}
blockquote + blockquote {
  margin-top: 0;
}
figcaption {
  font-family: "Plex Mono", serif, monospace;
  margin-top: ${lq * 2}px;
  font-size: ${line * 0.75 * bf}px;
  line-height: ${line * 0.75}px;
}
.info {
  background: #efefef;
  padding-left: 2ch;
  padding-right: 2ch;
  padding-top: ${lq * 2}px;
  padding-bottom: ${lq * 2}px;
  margin-bottom: ${line}px;
}
.info p:last-child {
  margin-bottom: 0;
}
img {
  display: block;
  position: relative;
  max-width: 100%;
  margin: 0 auto;
  page-break-inside: avoid;
}
code {
  font-size: 0.9em;
  line-height: 1.2;
  background: rgba(0,0,0,0.125);
  padding: 0 0.3em;
}
pre {
  font-size: 0.9em;
  line-height: 1.2;
  background: rgba(0,0,0,0.125);
  overflow-x: scroll;
  max-width: 100%;
  padding-left: 1ch;
  padding-right: 1ch;
  padding-top:0.625em;
  padding-bottom:0.625em;
}
pre code {
  background: transparent;
}

table {
  min-width: 100%;
  text-align: left;
  margin-top: ${lq * 2}px;
  font-size: ${line * 0.75 * bf}px;
  line-height: ${line * 0.675}px;
  border-collapse: collapse;
}
table, th, td {
  border: solid 1px black;
}
td {
  padding-left: 0.5ch;
  padding-right: 0.5ch;
  valign: top;
  vertical-align: top;
}
th {
  padding-left: 0.5ch;
  padding-right: 0.5ch;
  vertical-align: top;
  background: #efefef;
}
table ul, table ol {
  list-style-position: inside;
  padding-left: 0;
}
`

let sidebar_width = 32
let content_width = 64

function makeFonts() {
  return `
  @font-face {
    font-family: 'Plex Mono';
    src: url('fonts/IBMPlexMono-Regular.woff2') format('woff2'),
      url('fonts/IBMPlexMono-Regular.woff') format('woff');
    font-weight: normal;
    font-style: normal;
  }
  @font-face {
    font-family: 'Plex Mono';
    src: url('fonts/IBMPlexMono-Italic.woff2') format('woff2'),
      url('fonts/IBMPlexMono-Italic.woff') format('woff');
    font-weight: normal;
    font-style: italic;
  }
  @font-face {
    font-family: 'Plex Sans';
    src: url('fonts/IBMPlexSans-Regular.woff2') format('woff2'),
      url('fonts/IBMPlexSans-Regular.woff') format('woff');
    font-weight: normal;
    font-style: normal;
  }
  @font-face {
    font-family: 'Plex Sans';
    src: url('fonts/IBMPlexSans-Italic.woff2') format('woff2'),
      url('fonts/IBMPlexSans-Italic.woff') format('woff');
    font-weight: normal;
    font-style: italic;
  }
  @font-face {
    font-family: 'Plex Sans';
    src: url('fonts/IBMPlexSans-Bold.woff2') format('woff2'),
      url('fonts/IBMPlexSans-Bold.woff') format('woff');
    font-weight: bold;
    font-style: normal;
  }
  @font-face {
    font-family: 'Plex Sans';
    src: url('fonts/IBMPlexSans-BoldItalic.woff2') format('woff2'),
      url('fonts/IBMPlexSans-BoldItalic.woff') format('woff');
    font-weight: bold;
    font-style: italic;
  }
  `
}

function makeStyle() {
  return `<style type="text/css">
    ${makeFonts()}
    * {
      box-sizing: border-box;
    }
    html {
      background: #fff;
      font-family: "Plex Sans", serif, sans-serif;
      font-size: ${line * bf}px;
      line-height: ${line}px;
    }
    body {
      margin: 0;
    }
    .content {
      max-width: ${content_width}ch;
      padding-left: 2ch;
      padding-right: 2ch;
      margin: 0 auto;
      display: block;
      padding-bottom: ${line * 0}px;
    }
   p, ul, ol {
      margin: 0;
    }
    ul, ol {
      padding-left: 3ch;
    }
  p {
   // text-indent: 3ch;
}
    li p:first-child {
      text-indent: 0;
    }

    #pdf-logo {
      display: none;
    }

   hr {
      margin: 0;
      border-top-color: black;
      margin-top: -0.5px;
      margin-bottom: ${rlh - 0.5}px;
    }
  ${hcounter}
  a {
    color: inherit;
  }
  .table-of-contents {
    background: #efefef;
    position: fixed;
    left: 0;
    top: 0;
    width: ${sidebar_width}ch;
    height: 100vh;
    overflow-y: auto;
    background: #efefef;
      // background: rgba(230,230,230,0.85);
      //   backdrop-filter: blur(5px);
  }
  body {
    padding-left: ${sidebar_width}ch;
  }
  p:empty {
    display: none;
  }
  ul, ol {
  margin-bottom: 14px;
  }

  #report-iso {
    display: none;
  }

.table-of-contents {
    counter-reset: chapters;
}
 .table-of-contents ul {
    list-style: none;
    padding-left: 0;
    margin-bottom: 0;
  }
 .table-of-contents > ul {
  }
 .table-of-contents > ul > li {
    font-weight: bold;
  }
 .table-of-contents > ul > li {
    font-weight: bold;
   margin-bottom: ${rlh / 2}px;
  }

 .table-of-contents > ul > li > ul > li {
    font-weight: normal;
    font-style: normal;
    text-transform: none;
    letter-spacing: 0;
    margin-left: 0;
  }
 .table-of-contents > ul > li > ul > li > ul > li {
    font-weight: normal;
    font-style: italic;
  }
 .table-of-contents a {
    text-decoration: none;
  }
  .table-of-contents a:hover {
    text-decoration: underline;
  }
 sup {
  }
  .table-of-contents ul a {
    display: block;
    padding-left: 24px;
    text-indent: -8px;
    padding-right: 16px;
  }
  .table-of-contents ul li a.active {
    position: relative;
    background: #ddd;
    // text-decoration: line-through;
  }

 .table-of-contents > ul > li > ul > li > a {
    font-size: ${line * 0.9 * bf}px;
      line-height: ${line * 0.9}px;
    // padding-left: 4ch;
  }
  .table-of-contents > ul > li > ul > li > ul > li > a {
    padding-left: 5ch;
  }

h1 {
    counter-reset: chp;
}
h2 {
  position: relative;
  display: block;
  page-break-before: always;
  padding-top: ${rlh}px;
}
  .toc-desktop-hidden .table-of-contents {
    width: auto;
  }
  .toc-desktop-hidden #contents-label {
    display: none;
  }
  .toc-desktop-hidden .table-of-contents ul {
    display: none;
  }
  body.toc-desktop-hidden {
    padding-left: 5ch;
  }
  body:before {
    content: " ";
    height: ${line}px;
    width: 96ch;
    background: black;
    position: absolute;
    left: 0;
    top: 0;
    z-index: 999;
    display: none;
  }
    #toc-header {
      margin-top: ${lq * 2}px;
      margin-bottom: ${lq * 2}px;
      margin-left: 1ch;
      margin-right: 1ch;
    }

  @media screen and (max-width: 1028px) {
    h1 {
      font-size: ${line * 1.75 * hf}px;
      line-height: ${line * 1.75}px;
      font-weight: bold;
      margin-top: ${lq * 2}px;
      margin-bottom: ${lq * 2}px;
    }
    .table-of-contents ul li {
    }

    #toc-header {
      margin-top: ${lq}px;
      margin-bottom: ${lq}px;
    }

    body {
      padding-left: 0;
      padding-top: ${lq * 6}px;
    }
    .content {
        overflow-wrap: break-word;
        word-wrap: break-word;
    }
    #contents-label {
      display: none;
    }
    .table-of-contents {
      height: auto;
      width: 100%;
      z-index: 3;
    }
  body.toc-mobile-show .content:before {
      content: "";
      position: fixed;
      left: 0;
      top: 0;
      bottom: 0;
      right: 0;
      background: rgba(0,0,0,0.25);
      z-index: 2;
      border-top: solid ${line * 1.5}px #aaa;
    }

    .table-of-contents > ul {
      display: none;
    }
   body.toc-mobile-show {
      overflow: hidden;
    }
    body.toc-mobile-show #toc-header {
      margin-top: ${lq * 1}px;
      margin-bottom: ${lq * 1}px;
      position: relative;
    }
    body.toc-mobile-show .table-of-contents {
      width: ${sidebar_width}ch;
      height: 100vh;
      max-width: calc(100% - 4ch);
      overflow: auto;
    }
   body.toc-mobile-show .table-of-contents > ul {
      display: block;
      padding-bottom: ${line * 1}px;
      position: relative;
    }
    body.toc-mobile-show #contents-label {
      display: inline;
      position: relative;
    }
  }
}
</style>`
}

function makeJS() {
  return `<script>
    function inViewport(elem) {
      let bounding = elem.getBoundingClientRect();
      return (
        bounding.top >= 0 &&
        bounding.left >= 0 &&
        bounding.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
        bounding.right <= (window.innerWidth || document.documentElement.clientWidth)
      );
    };

    function setActive(target_id) {
      let selector = '.table-of-contents ul li a[href="#' + target_id + '"]'
      let link = document.querySelector(selector)
      if (link !== null) {
        link.className = 'active'
      }
    }

    window.addEventListener("load", (event) => {
      let headings = document.querySelectorAll('h2, h3');
      let links = document.querySelectorAll('.table-of-contents ul li a')

      observer = new IntersectionObserver((entry, observer) => {
        if (entry[0].intersectionRatio === 1) {
          for (let link of links) {
            link.className = ''
          }
          let target_id = entry[0].target.getAttribute('id')
          setActive(target_id)
        }
      }, { threshold: 1, rootMargin: "0px 0px -50% 0px" });

      let first = true
      for (let heading of headings) {
        if (first && inViewport(heading)) {
          setActive(heading.getAttribute('id'))
          first = false
        }
        observer.observe(heading);
      }

      document.querySelector('#toggle_contents').addEventListener('click', () => {
        let body = document.body
        if (window.innerWidth > 1027) {
          let hidden_class = "toc-desktop-hidden"
          if (body.className === hidden_class) {
            body.className = ''
          } else {
            body.className = hidden_class
          }
        } else {
          let show_class = "toc-mobile-show"
          if (body.className === show_class) {
            body.className = ''
          } else {
            body.className = show_class
          }
        }
      })

      for (let link of links) {
        link.addEventListener('click', (e) => {
          let href = e.target.getAttribute('href')
          let elem = document.getElementById(href.slice(1))
          window.scroll({
            top: elem.offsetTop - ${line},
            left: 0,
            behavior: 'smooth'
          })
          if (window.innerWidth < 1028) {
            document.body.className = ''
          }
          e.preventDefault() 
        })
      }

      document.querySelector('.content').addEventListener('click', () => {
        if (window.innerWidth < 1028) {
          document.body.className = ''
        }
      })
      document.querySelector('.table-of-contents').addEventListener('click', (e) => {
        e.stopPropagation()
      })

      let mediaQueryList = window.matchMedia("(max-width: 1028px)");
      function handleBreakpoint(mql) {
        // clear any left over toggle classes
        document.body.className = ''
      }
      mediaQueryList.addListener(handleBreakpoint);
    }, false);
  </script>`
}

function makeHead() {
  let title = 'Structural Time Series'
  let description =
    'An online research report on structural time series by Cloudera Fast Forward.'
  return `<head>
<meta charset="utf-8" />

<title>${title}</title>
<meta name="description" content="${description}" />

<meta property="og:title" content="${title}" /> 
<meta property="og:description" content="${description}" />
<meta property="og:image" content="https://structural-time-series.fastforwardlabs.com/figures/ff16-cover-splash.png" />
<meta property="og:url" content="https://structural-time-series.fastforwardlabs.com" />
<meta name="twitter:card" content="summary_large_image" />

<meta name="viewport" content="width=device-width" />
<link rel="icon" type="image/x-icon" href="favicon.ico" />

${makeStyle()}
${makeJS()}

<!-- Google Analytics -->
<script>
  (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
  (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
  m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
  })(window,document,'script','https://www.google-analytics.com/analytics.js','ga');

  ga('create', 'UA-157475426-9', 'auto');
  ga('send', 'pageview');

  window.addEventListener('load', function() {
    document.getElementById('report-pdf-download').addEventListener('click', function() {
      ga('send', {
        hitType: 'pageview',
        page: '//FF16-Structural_Time_Series-Cloudera_Fast_Forward.pdf'
      });
    });
  })

</script>
<!-- End Google Analytics -->
</head>`
}

function wrap(content) {
  return `<!DOCTYPE html>
    <html lang="en">
      ${makeHead()}
      <body>
        <div class="content" style="position: relative;">
          <div id="html-logo" style="margin-top: ${line}px; line-height: 0; display: flex;">
            <a href="https://www.cloudera.com/products/fast-forward-labs-research.html"><img alt="Cloudera Fast Forward" style="display: block; height: 14px; margin-bottom: 7px;" src='/figures/cloudera-fast-forward.png' /></a>
          </div>
          <div id="pdf-logo" style="margin-top: ${line}px; ">
            <a href="https://www.cloudera.com/products/fast-forward-labs-research.html">Cloudera Fast Forward</a>
          </div>
          ${content}
        </div>
      </body>
   </html>
  `
}

let filenames = fs.readdirSync(path.join(__dirname, 'src'))
// let filenames = ['00-frontmatter.md', '03-prototype.md'];
//
filenames = filenames.filter(function(file) {
  return path.extname(file).toLowerCase() === '.md'
})
console.log(filenames)

let report = ''
for (let f = 0; f < filenames.length; f++) {
  console.log(filenames[f])
  let content = fs.readFileSync(
    path.join(__dirname, 'src/') + filenames[f],
    'utf-8'
  )
  report += content + `\n`
}
let prehtml = wrap(md.render(report))

// wrap tables for mobile, not supposed to do this
html = prehtml.replace(
  /(<table[^>]*>(?:.|\n)*?<\/table>)/g,
  '<div style="width: 100%; overflow-x: auto;">$1</div>'
)

let write_index_to = path.join(__dirname, 'out/')
if (deploy_location === 'exp') {
  fs.mkdir(path.join(__dirname, 'exp'))
  fs.copySync(path.join(__dirname, 'out'), path.join(__dirname, 'exp'))
  write_index_to = path.join(__dirname, 'exp/')
}

fs.writeFileSync(write_index_to + 'index.html', html)
