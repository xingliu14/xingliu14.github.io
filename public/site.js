/* xingliu.me: small additions to pages that already work without scripts (search then goes to
   DuckDuckGo). ES5, no build step; each feature sits in its own try, so one failure never stops
   the rest. The contract is spec section 7. */
(function () {
  "use strict";
  var d = document, w = window, root = d.documentElement;
  function $(id) { return d.getElementById(id); }
  function each(list, fn) { for (var i = 0; i < list.length; i++) fn(list[i], i); }
  function trim(s) { return s.replace(/^\s+|\s+$/g, ""); }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return "&#" + c.charCodeAt(0) + ";"; }); }
  function onFrame(fn) { // at most once per frame
    var queued = false;
    return function () {
      if (queued) return;
      queued = true;
      (w.requestAnimationFrame || setTimeout)(function () { queued = false; fn(); });
    };
  }
  function el(tag, cls, text) { var e = d.createElement(tag); e.className = cls; if (text) e.textContent = text; return e; }

  // Polite live region for "Copied" (7.2).
  var live = el("p", "visually-hidden");
  live.id = "announce";
  live.setAttribute("aria-live", "polite");
  d.body.appendChild(live);
  function announce(msg) { live.textContent = ""; setTimeout(function () { live.textContent = msg; }, 50); }

  function copyText(text, btn) {
    function done() {
      announce("Copied to the clipboard");
      if (!btn.getAttribute("data-label")) btn.setAttribute("data-label", btn.textContent);
      btn.textContent = "Copied!";
      clearTimeout(btn.copyTimer);
      btn.copyTimer = setTimeout(function () { btn.textContent = btn.getAttribute("data-label"); }, 2000);
    }
    function fallback() { // a hidden textarea and execCommand: file:// and older browsers
      var back = d.activeElement, ta = d.createElement("textarea"), ok = false;
      ta.value = text;
      ta.setAttribute("readonly", "");
      ta.style.cssText = "position:fixed;top:0;left:-9999px";
      d.body.appendChild(ta);
      ta.select();
      try { ok = d.execCommand("copy"); } catch (e) { ok = false; }
      d.body.removeChild(ta);
      if (back && back.focus) back.focus();
      if (ok) done(); else announce("Could not copy. Select the text and copy it by hand.");
    }
    if (w.isSecureContext && navigator.clipboard) navigator.clipboard.writeText(text).then(done, fallback);
    else fallback();
  }

  // Case and width folding that keeps the length, so folded offsets fit the original.
  function fold(s) {
    var out = "", i, c, f;
    for (i = 0; i < s.length; i++) {
      c = s.charAt(i);
      f = (c.normalize ? c.normalize("NFKC") : c).toLowerCase();
      if (f.length !== 1) { f = c.toLowerCase(); if (f.length !== 1) f = c; }
      out += f;
    }
    return out;
  }

  // Search index: inline in the mockups (file:// cannot fetch), else /search.json, once.
  var index = null;
  function loadIndex() {
    if (!index) {
      index = new Promise(function (ok, no) {
        var inline = $("search-index");
        if (inline) ok(JSON.parse(inline.textContent));
        else fetch("/search.json", { credentials: "omit" }).then(function (r) { if (!r.ok) throw r.status; return r.json(); }).then(ok, no);
      }).then(function (docs) {
        each(docs, function (doc) {
          doc.x = doc.x || "";
          doc.ft = fold(doc.t);
          doc.fx = fold(doc.x);
          doc.fh = fold(doc.t + "\n" + doc.s + "\n" + doc.k + "\n" + doc.x);
        });
        return docs;
      });
      index.then(null, function () { index = null; }); // try again next time
    }
    return index;
  }
  function ddg(raw) { return esc("https://duckduckgo.com/?q=" + encodeURIComponent(raw) + "%20site%3Axingliu.me"); }
  // doc.h: a mockup's own file. A title's last two words stay on one line.
  function link(doc, cls) {
    return "<a" + (cls ? ' class="' + cls + '"' : "") + ' href="' + esc(doc.h || doc.u) + '"' + (doc.l === "zh-CN" ? ' lang="zh-CN"' : "") + ">" + esc(doc.t).replace(/ (?=\S+$)/, "\u00a0") + "</a>";
  }

  // 7.9 Colour menus: grey, white or night, remembered; Automatic follows the system.
  try {
    var menus = d.querySelectorAll("select.colours");
    each(menus, function (menu) {
      menu.value = root.getAttribute("data-theme") || "";
      menu.addEventListener("change", function () {
        var v = menu.value, tint = { grey: "#c0c0c0", white: "#ffffff", night: "#333333" }[v];
        if (tint) root.setAttribute("data-theme", v); else root.removeAttribute("data-theme");
        each(d.querySelectorAll('meta[name="theme-color"]'), function (m) {
          m.setAttribute("content", tint || (/dark/.test(m.getAttribute("media")) ? "#333333" : "#c0c0c0"));
        });
        d.querySelector('meta[name="color-scheme"]').setAttribute("content", v === "night" ? "dark" : tint ? "light" : "light dark");
        try { if (tint) localStorage.setItem("colours", v); else localStorage.removeItem("colours"); } catch (e) { /* private window */ }
        each(menus, function (m) { m.value = tint ? v : ""; });
      });
    });
  } catch (e) { /* colours stay as they are */ }

  // 7.3 Permalink: focus selects it; "Copy link" copies it.
  try {
    var permalink = $("permalink"), copyLink = d.querySelector(".copy-link");
    permalink.addEventListener("focus", function () { permalink.select(); });
    copyLink.addEventListener("click", function () { copyText(permalink.value, copyLink); });
  } catch (e) { /* not a post */ }

  // 7.4 Code bars: the language and a Copy button above each code well (room reserved by CSS).
  try {
    var NAMES = { ruby: "Ruby", php: "PHP", java: "Java", html: "HTML", console: "Console", css: "CSS", yaml: "YAML", python: "Python", javascript: "JavaScript", typescript: "TypeScript", bash: "Shell", sh: "Shell", shell: "Shell", zsh: "Shell", json: "JSON", c: "C", cpp: "C++", go: "Go", rust: "Rust", sql: "SQL", scala: "Scala", plaintext: "", text: "" };
    each(d.querySelectorAll("div.highlighter-rouge"), function (box) {
      var pre = box.querySelector("pre"), m = /(?:^|\s)language-(\S+)/.exec(box.className), lang = m ? m[1] : "text";
      var name = NAMES.hasOwnProperty(lang) ? NAMES[lang] : lang, bar = el("div", "code-bar"), btn = el("button", "small copy", "Copy");
      btn.type = "button";
      btn.setAttribute("aria-label", name ? "Copy " + name + " code" : "Copy code");
      btn.addEventListener("click", function () { copyText(pre.innerText.replace(/\n$/, ""), btn); });
      bar.appendChild(el("span", "lang", name));
      bar.appendChild(btn);
      box.insertBefore(bar, box.firstChild);
    });
  } catch (e) { /* code stays plain */ }

  // 7.5 "Scroll sideways" under a table only while it is wider than its box.
  try {
    var tables = d.querySelectorAll(".article .table-scroll"), hints = function () {
      each(tables, function (box) {
        var next = box.nextElementSibling, has = next && next.className === "scroll-hint";
        if (box.scrollWidth > box.clientWidth + 1) { if (!has) box.parentNode.insertBefore(el("p", "scroll-hint", "Scroll sideways for more columns \u2192"), next); }
        else if (has) box.parentNode.removeChild(next);
      });
    };
    if (tables.length) { hints(); w.addEventListener("resize", onFrame(hints)); }
  } catch (e) { /* tables still scroll */ }

  // 7.6 Search on /writing/: every term must match; titles first, then newest first.
  try {
    var form = $("search"), q = $("q"), status = $("search-status");
    if (form && q && status && !form.hasAttribute("data-go")) {
      var docs = null, results = status.parentNode;
      // Context around the match in the text, else in the description; the match in bold.
      var snippet = function (doc, query, terms) {
        var x = doc.x, fx = doc.fx, phrase = "", pass, i, cut;
        for (pass = 0; pass < 2 && !phrase; pass++) {
          if (pass) { x = doc.s; fx = fold(x); }
          phrase = fx.indexOf(query) !== -1 ? query : "";
          for (i = 0; !phrase && i < terms.length; i++) if (fx.indexOf(terms[i]) !== -1) phrase = terms[i];
        }
        if (!phrase) return esc(doc.s);
        var at = fx.indexOf(phrase), end = at + phrase.length, from = Math.max(0, at - 70), to = Math.min(x.length, end + 90);
        // Cut at a word boundary when one is close (English); Chinese has none.
        if (from > 0 && (cut = x.slice(from, at).search(/\s/)) !== -1 && cut < 20) from += cut + 1;
        if (to < x.length && (cut = x.slice(end, to).lastIndexOf(" ")) !== -1 && to - end - cut < 20) to = end + cut;
        if (/[\udc00-\udfff]/.test(x.charAt(from))) from++; // never split a surrogate pair
        if (/[\ud800-\udbff]/.test(x.charAt(to - 1))) to--;
        return (from ? "\u2026" : "") + esc(x.slice(from, at)) + "<b>" + esc(x.slice(at, end)) + "</b>" + esc(x.slice(end, to)) + (to < x.length ? "\u2026" : "");
      };
      var setUrl = function (url) { try { history.replaceState(null, "", url); } catch (e) { /* file:// */ } };
      // The count is a live region: while typing, it changes once the reader pauses.
      var timer, say = function (html, typing) {
        clearTimeout(timer);
        timer = setTimeout(function () { status.innerHTML = html; }, typing ? 400 : 0);
      };
      var run = function (e) {
        var raw = trim(q.value), old = results.querySelectorAll(".hits, .other");
        each(old, function (o) { results.removeChild(o); });
        if (!raw) { say(""); setUrl(location.pathname + "#search"); return; }
        if (!docs) {
          status.textContent = "Loading the search index\u2026";
          loadIndex().then(function (all) { docs = all; run(); }, function () {
            status.innerHTML = 'The search index could not be loaded. <a href="' + ddg(raw) + '">Search xingliu.me with DuckDuckGo</a> instead.';
          });
          return;
        }
        var query = fold(raw), terms = query.split(/\s+/), first = [], rest = [], html = "";
        each(docs, function (doc) {
          var all = true, title = true;
          each(terms, function (t) { all = all && doc.fh.indexOf(t) !== -1; title = title && doc.ft.indexOf(t) !== -1; });
          if (all) (title ? first : rest).push(doc);
        });
        var hits = first.concat(rest), n = hits.length, said = "\u201c" + esc(raw) + "\u201d";
        say(n ? "<b>" + n + (n === 1 ? " post</b> matches " : " posts</b> match ") + said + ":" : "No posts match " + said + ".", e && e.type === "input");
        // \u2060 (word joiner) keeps the chip with the title's last words.
        each(hits, function (h) {
          var p = h.d.split("-"), zh = h.l === "zh-CN" ? ' lang="zh-CN"' : "";
          html += "<li>" + link(h, "t") + (zh ? '\u2060<span class="zh" lang="zh-CN" title="Written in Chinese">\u4e2d\u6587</span>' : "") +
            '\n<p class="snip"' + zh + ">" + snippet(h, query, terms) + '</p>\n<p class="where"><span class="u">xingliu.me' + esc(h.u) +
            "</span>\u00a0\u00b7 " + MONTHS[+p[1] - 1] + " " + +p[2] + ", " + p[0] + "\u00a0\u00b7 " + esc(h.k) + "</p></li>\n";
        });
        var other = el("p", "other");
        other.innerHTML = n ? 'Not what you wanted? <a href="' + ddg(raw) + '">Search xingliu.me with DuckDuckGo</a>.' : 'Try fewer words, or <a href="' + ddg(raw) + '">search xingliu.me with DuckDuckGo</a>.';
        results.appendChild(other);
        if (n) { var ol = el("ol", "hits"); ol.innerHTML = html; results.insertBefore(ol, other); }
        setUrl("?q=" + encodeURIComponent(raw) + "#search");
      };
      var MONTHS = "January February March April May June July August September October November December".split(" ");
      q.addEventListener("focus", function () { loadIndex().then(null, function () {}); });
      q.addEventListener("input", run);
      form.addEventListener("submit", function (e) { e.preventDefault(); run(); });
      // Arrival: a ?q= link (footer forms send one), or a value the browser kept.
      var m = /[?&]q=([^&#]*)/.exec(location.search);
      if (m) { try { q.value = decodeURIComponent(m[1].replace(/\+/g, " ")); } catch (e) { q.value = m[1]; } }
      if (trim(q.value)) run();
      // Focus for #search, and again after load: Chrome's fragment scroll clears it.
      var focus = function () { if (location.hash === "#search" && (d.activeElement === d.body || !d.activeElement)) q.focus(); };
      focus();
      w.addEventListener("load", function () { setTimeout(focus, 0); });
      w.addEventListener("hashchange", function () { q.focus(); });
    }
  } catch (e) { /* the form still goes to DuckDuckGo */ }

  // 7.7 The footer and Not Found forms search on /writing/.
  try {
    each(d.querySelectorAll("form[data-go]"), function (f) {
      f.addEventListener("submit", function (e) {
        var v = trim(f.querySelector('input[name="q"]').value);
        e.preventDefault();
        location.href = f.getAttribute("data-go") + (v ? "?q=" + encodeURIComponent(v) : "") + "#search";
      });
    });
  } catch (e) { /* they go to DuckDuckGo */ }

  // 7.8 Not Found: name the path, pre-fill the search, suggest the likely post.
  try {
    var missing = $("missing"), path = /^https?:$/.test(location.protocol) ? location.pathname : d.body.getAttribute("data-demo-path");
    if (missing && path) {
      try { path = decodeURI(path); } catch (e) { /* keep it raw */ }
      missing.textContent = "The requested URL ";
      missing.appendChild(el("code", "", path));
      missing.appendChild(d.createTextNode(" was not found on this server."));
      var segs = path.split("/"), last = "", words = [], i;
      for (i = segs.length - 1; i >= 0 && !last; i--) if (segs[i] !== "index.html") last = segs[i];
      each(last.replace(/\.html?$/i, "").split(/[-_.\s]+/), function (word) { if (word && !/^\d+$/.test(word)) words.push(word); });
      if (!$("q404").value) $("q404").value = words.join(" ");
      if (last) loadIndex().then(function (all) {
        var want = path.toLowerCase(), seg = last.toLowerCase(), hit = null, why;
        if (!/\.[a-z0-9]+$/.test(seg) && !/\/$/.test(want)) want += "/";
        each(all, function (doc) { if (!hit && doc.u.toLowerCase() === want) { hit = doc; why = "Addresses on this server are case-sensitive."; } });
        each(all, function (doc) { if (!hit && doc.u.toLowerCase().split("/").slice(-2)[0] === seg) { hit = doc; why = "It lives at a dated address."; } });
        if (!hit) return;
        $("didyoumean").innerHTML = "Did you mean " + link(hit) + '? <span class="why">' + why + "</span>";
        $("didyoumean").hidden = false;
      }, function () { /* no suggestion */ });
    }
  } catch (e) { /* the generic sentence stays */ }

  // 7.10 NEW! expires on the reader's clock.
  try {
    var now = new Date(), today = now.getFullYear() + "-" + ("0" + (now.getMonth() + 1)).slice(-2) + "-" + ("0" + now.getDate()).slice(-2);
    each(d.querySelectorAll(".new[data-until]"), function (tag) { if (tag.getAttribute("data-until") <= today) tag.parentNode.removeChild(tag); });
  } catch (e) { /* the tag stays as built */ }

  // 7.11 Contents: links to it open the box; wide screens mark the section in view.
  try {
    var toc = $("toc"), box = toc.querySelector("details"), marks = [], here = null;
    var rail = w.matchMedia && w.matchMedia("(min-width: 77.5em)");
    each(d.querySelectorAll('a[href="#toc"]'), function (a) { a.addEventListener("click", function () { box.open = true; }); });
    each(toc.querySelectorAll('a[href^="#"]'), function (a) {
      var id = a.getAttribute("href").slice(1);
      try { id = decodeURIComponent(id); } catch (e) { /* raw */ }
      if ($(id)) marks.push({ a: a, h: $(id) });
    });
    var mark = onFrame(function () {
      var cur = null, line = w.innerHeight * 0.35, i;
      for (i = 0; rail && rail.matches && i < marks.length && marks[i].h.getBoundingClientRect().top <= line; i++) cur = marks[i];
      if (cur === here) return;
      if (here) here.a.removeAttribute("aria-current");
      here = cur;
      if (!cur) return;
      cur.a.setAttribute("aria-current", "location");
      // When the list scrolls, keep the marked line in view.
      var list = cur.a.parentNode.parentNode, top = cur.a.offsetTop - list.offsetTop;
      if (top < list.scrollTop || top > list.scrollTop + list.clientHeight - 30) list.scrollTop = top - list.clientHeight / 2;
    });
    w.addEventListener("scroll", mark, { passive: true });
    w.addEventListener("resize", mark);
    if (rail && rail.addListener) rail.addListener(mark);
    mark();
    // Print the box open.
    var was;
    w.addEventListener("beforeprint", function () { was = box.open; box.open = true; });
    w.addEventListener("afterprint", function () { box.open = was; });
  } catch (e) { /* the box still opens and closes */ }

  // 7.12 Visitor map: after load, when idle or the well is near. Only over http(s), and only
  // where the well holds the placeholder (the mockups hold a still).
  try {
    var well = $("visitor-map"), wait = well.querySelector(".mm-wait");
    if (wait && /^https?:$/.test(location.protocol)) {
      var start = function () {
        var loaded = false, io = null, load = function () {
          if (loaded) return;
          loaded = true;
          if (io) io.disconnect();
          well.removeChild(wait);
          var s = d.createElement("script");
          s.id = "mapmyvisitors";
          s.async = true;
          s.src = well.getAttribute("data-src");
          well.appendChild(s);
        };
        wait.hidden = false;
        if (w.IntersectionObserver) {
          io = new IntersectionObserver(function (seen) { each(seen, function (s) { if (s.isIntersecting) load(); }); }, { rootMargin: "400px 0px" });
          io.observe(well);
        }
        if (w.requestIdleCallback) w.requestIdleCallback(load, { timeout: 2000 }); else setTimeout(load, 1500);
      };
      if (d.readyState === "complete") start(); else w.addEventListener("load", start);
    }
  } catch (e) { /* the placeholder stays */ }
})();
