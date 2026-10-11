import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";
import { ComfyWidgets } from "/scripts/widgets.js";
import { SearchableCombo } from "./SearchableCombo.js";

async function onLoraChanged(node, lora_name) {
    try {
        const response = await api.fetchApi("/on_demand_loader/lora_changed", {
            method: "POST",
            body: JSON.stringify({ lora_name }),
            headers: {
                "Content-Type": "application/json",
            },
        });
        if (response.status !== 200) {
            console.error(`[ComfyUI-OnDemand-Loaders] Failed to notify backend of lora change: ${response.status}`);
        } else {
			const loraInfo = await response.json();
			addOrUpdateLoraInfoWidgets(node, loraInfo);
		}
    } catch (e) {
        console.error("[ComfyUI-OnDemand-Loaders] Failed to notify backend of lora change", e);
    }
}

async function onFilterChanged(node, filter) {
    try {
        const response = await api.fetchApi("/on_demand_loader/filter_changed", {
            method: "POST",
            body: JSON.stringify({ filter }),
            headers: {
                "Content-Type": "application/json",
            },
        });
        if (response.status !== 200) {
            console.error(`[ComfyUI-OnDemand-Loaders] Failed to notify backend of filter change: ${response.status}`);
        } else {
			const filteredLoras = await response.json();
			const w = node.widgets.find((w) => w.name === "lora_name")
			w.options.values = filteredLoras.map(lora => lora.name);
		}
    } catch (e) {
        console.error("[ComfyUI-OnDemand-Loaders] Failed to notify backend of filter change", e);
    }
}

function addOrUpdateLoraInfoWidgets(node, loraInfo) {
	if (loraInfo.name !== "None") {

		if (loraInfo.author) {
			addOrUpdateWidget(node, "author", loraInfo.author);
		}

		if (loraInfo.trigger_words && loraInfo.trigger_words.length > 0) {
			addOrUpdateWidget(node, "trigger_words", loraInfo.trigger_words.join(', '), "STRING", true);
		}

		if (loraInfo.id) {
			const urlValue = `[Civitai Model URL](https://civitai.com/models/${loraInfo.id})`;
			addOrUpdateWidget(node, "url", urlValue, "MARKDOWN", false);
		}

		if (loraInfo.base_model) {
			addOrUpdateWidget(node, "base_model", loraInfo.base_model);
		}
	} else {
		hideWidgetIfExists(node, "author");
		hideWidgetIfExists(node, "trigger_words");
		hideWidgetIfExists(node, "url");
		hideWidgetIfExists(node, "base_model");
	}
	node.computeSize();
	if (typeof node.setDirtyCanvas === 'function') {
      node.setDirtyCanvas(true, true);
    }
}

function hideWidgetIfExists(node, widgetName) {
	const w = node.widgets.find((w) => w.name === widgetName);
	if (w) {
		w.hidden = true;
	}
}

function addOrUpdateWidget(node, widgetName, value, type = "STRING", multiline = false) {
	const w = node.widgets.find((w) => w.name === widgetName);
	if (!w) {
		const newW = ComfyWidgets[type](node, widgetName, ["STRING", { multiline: multiline }], app).widget;
		if (newW.inputEl) {
			newW.inputEl.readOnly = true;
			newW.inputEl.style.opacity = 0.8;
		}
		newW.value = value;
	} else {
		w.value = value
		w.hidden = false;
	}
}

// Config-based loaders: each `url` widget is tied to a model name widget and a config.json list
// through the ondemand_* options of its input definition (see _url_input in nodes.py).
function getUrlLinks(nodeData) {
	return Object.entries(nodeData.input?.optional ?? {})
		.filter(([, spec]) => spec?.[1]?.ondemand_config_key)
		.map(([urlInput, spec]) => ({
			urlInput,
			nameInput: spec[1].ondemand_name_input,
			configKey: spec[1].ondemand_config_key,
		}));
}

function getLinkWidgets(node, link) {
	return {
		nameWidget: node.widgets?.find((w) => w.name === link.nameInput),
		urlWidget: node.widgets?.find((w) => w.name === link.urlInput),
	};
}

async function populateUrl(node, link, { onlyIfEmpty = false } = {}) {
	const { nameWidget, urlWidget } = getLinkWidgets(node, link);
	if (!nameWidget || !urlWidget) return;
	const name = nameWidget.value;
	try {
		const params = new URLSearchParams({ key: link.configKey, name });
		const response = await api.fetchApi(`/on_demand_loader/model_url?${params}`);
		if (response.status !== 200) {
			console.error(`[ComfyUI-OnDemand-Loaders] Failed to fetch url for '${name}': ${response.status}`);
			return;
		}
		const { url } = await response.json();
		// Ignore stale responses, and keep URLs already saved in a loaded workflow
		if (nameWidget.value !== name || (onlyIfEmpty && urlWidget.value)) return;
		// A name that is not in config.json yet keeps the URL the user entered
		if (url || name === "None") {
			urlWidget.value = url;
			node.setDirtyCanvas?.(true, true);
		}
	} catch (e) {
		console.error(`[ComfyUI-OnDemand-Loaders] Failed to fetch url for '${name}'`, e);
	}
}

// The values array is shared by all nodes of a type; `local` adds a value to this widget only,
// for names that are not saved in config.json yet.
function addComboValue(widget, value, { local = false } = {}) {
	const values = widget.options?.values;
	if (!Array.isArray(values) || values.includes(value)) return;
	if (local) {
		widget.options = { ...widget.options, values: [...values, value] };
	} else {
		values.push(value);
	}
}

async function promptNewModelName(node, link) {
	const { nameWidget, urlWidget } = getLinkWidgets(node, link);
	if (!nameWidget || !urlWidget) return;

	const result = await showNewModelDialog(`New ${link.nameInput}`);
	if (!result) return;

	addComboValue(nameWidget, result.name, { local: true });
	nameWidget.value = result.name;
	urlWidget.value = result.url;
	node.setDirtyCanvas?.(true, true);
}

// Modal asking for a model name and download URL; resolves to { name, url }, or null if cancelled.
function showNewModelDialog(title) {
	return new Promise((resolve) => {
		const dialog = document.createElement("dialog");
		dialog.style.cssText = `
			padding: 20px; width: min(480px, calc(100vw - 32px)); box-sizing: border-box;
			border: 1px solid var(--border-color, #444); border-radius: 8px;
			background: var(--comfy-menu-bg, #202020); color: var(--fg-color, #ddd);
			font: 14px sans-serif;`;
		dialog.innerHTML = `
			<form method="dialog" style="display: flex; flex-direction: column; gap: 12px; margin: 0;">
				<div style="font-size: 16px; font-weight: 600;"></div>
				<div style="color: var(--descrip-text, #999);">The model is added to config.json and downloaded when the workflow runs.</div>
				<label style="display: flex; flex-direction: column; gap: 4px;">Name
					<input name="name" required autocomplete="off" placeholder="Model name">
				</label>
				<label style="display: flex; flex-direction: column; gap: 4px;">URL
					<input name="url" required autocomplete="off" placeholder="https://civitai.com/api/download/models/... or https://huggingface.co/.../resolve/...">
				</label>
				<div style="display: flex; justify-content: flex-end; gap: 8px;">
					<button type="button" value="cancel">Cancel</button>
					<button type="submit" value="ok">Add model</button>
				</div>
			</form>`;
		dialog.querySelector("div").textContent = title;
		for (const el of dialog.querySelectorAll("input, button")) {
			el.style.cssText = `
				padding: 6px 10px; border-radius: 6px; font: inherit;
				border: 1px solid var(--border-color, #444);
				background: var(--comfy-input-bg, #2a2a2a); color: var(--input-text, #ddd);`;
		}

		const form = dialog.querySelector("form");
		let result = null;
		form.addEventListener("submit", () => {
			result = { name: form.elements.name.value.trim(), url: form.elements.url.value.trim() };
		});
		dialog.querySelector('button[value="cancel"]').addEventListener("click", () => dialog.close());
		// Whitespace-only values pass `required`; reject them like empty ones
		for (const input of form.querySelectorAll("input")) {
			input.addEventListener("input", () => input.setCustomValidity(input.value.trim() ? "" : "Required"));
		}
		dialog.addEventListener("close", () => {
			dialog.remove();
			resolve(result);
		});

		document.body.appendChild(dialog);
		dialog.showModal();
		form.elements.name.focus();
	});
}

function setupUrlWidgets(nodeType, links) {
	const onNodeCreated = nodeType.prototype.onNodeCreated;
	nodeType.prototype.onNodeCreated = function () {
		const r = onNodeCreated?.apply(this, arguments);
		for (const link of links) {
			const { nameWidget } = getLinkWidgets(this, link);
			if (!nameWidget) continue;

			const callback = nameWidget.callback;
			const node = this;
			nameWidget.callback = function () {
				const r = callback?.apply(this, arguments);
				populateUrl(node, link);
				return r;
			};

			const label = links.length > 1 ? `➕ New ${link.nameInput}` : "➕ New model";
			this.addWidget("button", label, "", () => promptNewModelName(this, link), { serialize: false });

			populateUrl(this, link, { onlyIfEmpty: true });
		}
		return r;
	};

	const onConfigure = nodeType.prototype.onConfigure;
	nodeType.prototype.onConfigure = function () {
		const r = onConfigure?.apply(this, arguments);
		for (const link of links) {
			const { nameWidget } = getLinkWidgets(this, link);
			// Names saved in the workflow but not in config.json yet must stay selectable
			if (nameWidget) addComboValue(nameWidget, nameWidget.value, { local: true });
			populateUrl(this, link, { onlyIfEmpty: true });
		}
		return r;
	};
}

// When a run adds a model to config.json, offer it in the dropdowns of matching nodes
api.addEventListener("ondemand_loaders.config_updated", ({ detail }) => {
	for (const node of app.graph?.nodes ?? []) {
		for (const link of node.constructor.ondemandUrlLinks ?? []) {
			if (link.configKey !== detail.key) continue;
			const { nameWidget } = getLinkWidgets(node, link);
			if (nameWidget) addComboValue(nameWidget, detail.name);
		}
	}
});



app.registerExtension({
	name: "comfy.francarl.onDemandLoader",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		const urlLinks = getUrlLinks(nodeData);
		if (urlLinks.length > 0) {
			nodeType.ondemandUrlLinks = urlLinks;
			setupUrlWidgets(nodeType, urlLinks);
		}

		if (nodeData.name === "OnDemandCivitaiLikedLoraLoader") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				onNodeCreated?.apply(this, arguments);

				const loraNameWidget = this.widgets.find((w) => w.name === "lora_name");

				if (loraNameWidget) {

					loraNameWidget.callback = (value) => {
						onLoraChanged(this, value);
					};

					loraNameWidget.searchableCombo = new SearchableCombo();

					loraNameWidget.onClick = (e) => {
						console.log("[ComfyUI-OnDemand-Loaders] Lora Name:", loraNameWidget.value);

						loraNameWidget.searchableCombo.show(loraNameWidget.options.values, {
							event: e.e,
							title: 'Select Preset',
							currentMode: 'list',
							initialExpanded: false,
							onExpandedChange: (isExpanded) => {
								console.log('[ComfyUI-OnDemand-Loaders] SearchableCombo expanded:', isExpanded);
							},
							callback: (selectedItem) => {
								loraNameWidget.value = selectedItem;
								onLoraChanged(this, selectedItem);
							}
						});
					};
				}

				const filterWidget = this.widgets.find((w) => w.name === "filter");
				if (filterWidget) {
					filterWidget.callback = (value) => {
						onFilterChanged(this, value);
					};
				}
				
				this.addWidget("button", "ℹ️ Lora Info", "", () => {
					window.showSelectedLoraInfo(this, loraNameWidget);
				}, { serialize: false });
				
			};
		}
	},
	async setup() {
        window.showSelectedLoraInfo = async (node, widget) => {
            onLoraChanged(node, widget.value);
        };
    }
});