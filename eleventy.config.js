import markdownIt from "markdown-it";
import markdownItAnchor from "markdown-it-anchor";
import tocPlugin from "eleventy-plugin-toc";


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
        return `
            <figure>
                <img src="${src}" alt="${altTextFixed}">${captionHtml}
            </figure>
        `;
    });

    eleventyConfig.addShortcode("footnote-ref", (id) => {
        let footnoteEntry = footnoteMap.get(id);

        if (typeof footnoteEntry === "undefined") {
            footnoteEntry = [1 + footnoteMap.size, 1]; // [footnoteIdx, numReferences]
            footnoteMap.set(id, footnoteEntry);
        } else {
            footnoteEntry[1]++;
        }

        return `<a
            id='footnote-${id}-ref-${footnoteEntry[1]-1}'
            href='#footnote-${id}'
            class='footnote-ref'
        >[${footnoteEntry[0]}]</a>`;
    });

    eleventyConfig.addShortcode("footnote-content", (id, text) => {
        const footnoteEntry = footnoteMap.get(id);
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

    eleventyConfig.addFilter("postDate", (dateObj) =>
        postDateFormatter.format(dateObj));
};
