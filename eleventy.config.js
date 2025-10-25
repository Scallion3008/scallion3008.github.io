import markdownIt from "markdown-it";
import markdownItAnchor from "markdown-it-anchor";
import tocPlugin from "eleventy-plugin-toc";
import { v4 as uuidv4 } from "uuid";


const postDateFormatter = new Intl.DateTimeFormat("en-US", {
    dateStyle: "medium",
});


export default (eleventyConfig) => {
    const footnoteMap = new Map();

    eleventyConfig.setLibrary("md", markdownIt({ html: true })
        .use(markdownItAnchor, {
            permalink: markdownItAnchor.permalink.headerLink(),
        }));

    eleventyConfig.addPlugin(tocPlugin, { ul: true });
    
    eleventyConfig.addPassthroughCopy("fonts");
    eleventyConfig.addPassthroughCopy("media");
    eleventyConfig.addPassthroughCopy("styles");
    eleventyConfig.addPassthroughCopy("scripts");

    eleventyConfig.addShortcode("figure", (src, caption, altText = caption) => {
        // Remove HTML tags from alt text in case caption reused as alt text
        const altTextFixed = altText.replace(/<.*>/, " ");

        const captionHtml = caption ? `<figcaption>${caption}</figcaption>` : "";
        return `<figure><img src="${src}" alt="${altTextFixed}">${captionHtml}</figure>`;
    });

    eleventyConfig.addShortcode("footnote-ref", function (id) {
        let thisPageFootnotes = footnoteMap.get(this.page.url);
        if (typeof thisPageFootnotes === "undefined") {
            thisPageFootnotes = new Map();
            footnoteMap.set(this.page.url, thisPageFootnotes);
        }

        let footnoteEntry = thisPageFootnotes.get(id);
        if (typeof footnoteEntry === "undefined") {
            footnoteEntry = [1 + thisPageFootnotes.size, 1]; // [footnoteIdx, numReferences]
            thisPageFootnotes.set(id, footnoteEntry);
        } else {
            footnoteEntry[1]++;
        }

        return `<a
            id='footnote-${id}-ref-${footnoteEntry[1]-1}'
            href='#footnote-${id}'
            class='footnote-ref'
        >[${footnoteEntry[0]}]</a>`;
    });

    eleventyConfig.addShortcode("footnote-content", function (id, text) {
        const thisPageFootnotes = footnoteMap.get(this.page.url);
        if (typeof thisPageFootnotes === "undefined") {
            return "";
        }

        const footnoteEntry = thisPageFootnotes.get(id);
        if (typeof footnoteEntry === "undefined") {
            return "";
        }

        return `
            <p id='footnote-${id}' class='footnote-content'>
                <span class='footnote-id'>[${footnoteEntry[0]}]</span>
                <span class='footnote-backlinks'>${
                    [...Array(footnoteEntry[1]).keys().map(
                        (_, i) => `<a href='#footnote-${id}-ref-${i}'>^ </a>`
                    )].join('')
                }</span>
                <span>${text}</span>
            </p>
        `;
    });

    eleventyConfig.addPairedShortcode("panel-toggle", (content, toggleName, ...names) => {
        const panelId = uuidv4();

        return `<div class="panel-toggle">
<form><span>${toggleName}</span><div>${names.map((name, i) => `
<input type="radio" id="panel-toggle-${panelId}-${i}" name="panel" value="${i}" ${i == 0 ? "checked" : ""}>
<label for="panel-toggle-${panelId}-${i}">${name}</label>
`).join("")}</div>
</form><div class="panel-toggle-panels">${content}</div></div>`;
    });

    eleventyConfig.addPairedShortcode("panel", 
        (content) => `<div class="panel">${content}</div>`);

    eleventyConfig.addFilter("postDate", (dateObj) =>
        postDateFormatter.format(dateObj));
};
