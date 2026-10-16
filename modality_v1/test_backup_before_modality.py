"""Regression guards for preserving the previous audited diagnostic phase."""
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest
import zipfile
import backup_before_modality as m


class PreviousDiagnosticBackupTests(unittest.TestCase):
    def test_missing_binary_artifact_is_refused_with_real_audited_metadata(self):
        project=m.HERE.parents[1]
        archive=project/'项目记录/归档/20261015/fatigue-diagnostics-v1-audit-20261015-145526-81211.zip'
        if not archive.exists():self.skipTest('Local previous-audit ZIP absent')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);diag=root/'diag';diag.mkdir()
            anchors=m.b.read(m.HERE/'previous_diagnostic_anchors.json')
            with zipfile.ZipFile(archive) as z:
                for name in anchors:
                    target=diag/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
            # Deliberately omitted .pt/.npy are not an acceptable complete backup.
            with patch.object(m.b,'collect',return_value=({}, {}, (root/'old',))),self.assertRaises(FileNotFoundError):
                m.collect(root/'old',root/'e0',root/'code',diag)

    def test_previous_active_writer_is_refused_before_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'writer.lock').write_text('existing writer',encoding='utf-8')
            with patch.object(m.b,'collect',return_value=({}, {}, (root/'old',))),patch.object(m.d,'status_report') as status:
                with self.assertRaises(ValueError):m.collect(root/'old',root/'e0',root/'code',root)
                status.assert_not_called()
            self.assertEqual((root/'writer.lock').read_text(encoding='utf-8'),'existing writer')

    def test_changed_audited_anchor_cannot_be_backed_as_the_same_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);diag=root/'diag';diag.mkdir();(diag/'ledger.json').write_text('{}',encoding='utf-8')
            with patch.object(m.b,'collect',return_value=({}, {}, (root/'old',))),patch.object(m.d,'status_report') as status:
                with self.assertRaisesRegex(ValueError,'anchor changed'):m.collect(root/'old',root/'e0',root/'code',diag)
                status.assert_not_called()


if __name__=='__main__':unittest.main()
