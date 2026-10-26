// CNN Vision - monthly check (Google Apps Script, runs on your Google account at script.google.com).
// On the 1st of every month, after the free tier resets and the site has had time to reopen by itself,
// it checks that the site answers. If it doesn't (closed by the charge cut-off, or something went wrong),
// it emails you a reminder to run "Restart CNN Vision" on your desktop.
// Setup: paste this into a new Apps Script project, fill in EMAIL, then run setup() once and allow access.

const SITE = 'https://cnn-vision-134174130288.us-central1.run.app';
const EMAIL = 'you@example.com';

function setup() {
  ScriptApp.getProjectTriggers().forEach(t => ScriptApp.deleteTrigger(t));
  // 1st of the month at 15:00-16:00 in this script's time zone: after the 09:00 UTC reopen job
  ScriptApp.newTrigger('checkSite').timeBased().onMonthDay(1).atHour(15).create();
  checkSite();
  MailApp.sendEmail(EMAIL, 'CNN Vision monthly check is set up',
    'You will get an email here if ' + SITE + ' is still offline after the reset on the 1st of a month.');
}

function checkSite() {
  let status = 'no answer';
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      // The first request may wake the server up, which can take up to a minute
      const code = UrlFetchApp.fetch(SITE + '/api/health', {muteHttpExceptions: true}).getResponseCode();
      if (code === 200) return;
      status = 'HTTP ' + code + (code === 403 ? ' (closed to the public)' : '');
    } catch (e) {
      status = String(e);
    }
    Utilities.sleep(20000);
  }
  MailApp.sendEmail(EMAIL, 'CNN Vision is offline - restart it from your desktop',
    'CNN Vision (' + SITE + ') did not come back online after the monthly reset: ' + status + '.\n\n' +
    'It was probably closed by the charge cut-off, which keeps it closed until you restart it yourself.\n\n' +
    'To restart it: double-click "Restart CNN Vision" on your desktop.\n' +
    'It checks this month\'s usage, opens the site again and tells you when it is back.');
}
